import os, math, torch, soundfile as sf, librosa, warnings, numpy as np, onnxruntime as ort, logging, contextlib, io
from types import SimpleNamespace
from torch import nn
from torch.nn import functional as F
from transformers.modeling_outputs import MoeCausalLMOutputWithPast
from transformers import SiglipImageProcessor, SiglipVisionModel, logging as hf_logging, AutoModelForCausalLM
from .model_minimind import *


class OmniConfig(MiniMindConfig):
    model_type = "minimind-o"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_talker_hidden_layers = kwargs.get("num_talker_hidden_layers", 4)
        self.talker_hidden_size = kwargs.get("talker_hidden_size", 768)
        self.audio_ids = kwargs.get("audio_ids", [151656]) # "<|video_pad|>" token id
        self.audio_special_token = kwargs.get("audio_special_token", "<|video_pad|>")
        self.audio_hidden_size = kwargs.get("audio_hidden_size", 512)
        self.audio_vocab_size = kwargs.get("audio_vocab_size", 2112)
        self.audio_pad_token = kwargs.get("audio_pad_token", 2049)
        self.audio_stop_token = kwargs.get("audio_stop_token", 2050)
        self.audio_spk_token = kwargs.get("audio_spk_token", 2051)
        self.spk_emb_size = kwargs.get("spk_emb_size", 192)
        self.think_end_ids = kwargs.get("think_end_ids", [151668]) # </think>
        self.image_ids = kwargs.get("image_ids", [151655]) # "<|image_pad|>" token id
        self.image_special_token = kwargs.get("image_special_token", "<|image_pad|>")
        self.image_hidden_size = kwargs.get("image_hidden_size", 768)
        self.image_token_len = kwargs.get("image_token_len", 64)
        self.bridge_layer = kwargs.get("bridge_layer", self.num_hidden_layers // 2 - 1)

class MMAudioProjector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.input_norm = nn.LayerNorm(in_dim)
        self.linear = nn.Linear(in_dim, out_dim)
        self.mlp = nn.Sequential(
            nn.Linear(out_dim, 2 * out_dim),
            nn.GELU(),
            nn.Linear(2 * out_dim, out_dim, bias=False),
        )
    def forward(self, x):
        x = self.linear(self.input_norm(x))
        x = x + self.mlp(x)
        return x

class MMVisionProjector(nn.Module):
    def __init__(self, in_dim, out_dim, source_tokens=64, target_tokens=64):
        super().__init__()
        self.input_norm = nn.LayerNorm(in_dim)
        self.linear = nn.Linear(in_dim, out_dim)
        self.mlp = nn.Sequential(
            nn.Linear(out_dim, 2 * out_dim),
            nn.GELU(),
            nn.Linear(2 * out_dim, out_dim, bias=False),
        )
    def forward(self, x):
        x = self.linear(self.input_norm(x))
        x = x + self.mlp(x)
        return x

class TalkerHead(nn.Module):
    def __init__(self, in_features, out_features, num_layers=8, rank=256):
        super().__init__()
        self.num_layers = num_layers
        self.base = nn.Linear(in_features, out_features, bias=False)
        self.adapters = nn.ModuleList([nn.Sequential(nn.Linear(in_features, rank, bias=False), nn.GELU(), nn.Linear(rank, out_features, bias=False)) for _ in range(num_layers)])
    def forward(self, x):
        base_out = self.base(x)
        return [base_out + adapter(x) for adapter in self.adapters]


class TalkerEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, num_layers=8, rank=256):
        super().__init__()
        self.num_layers = num_layers
        self.base = nn.Embedding(num_embeddings, embedding_dim)
        self.adapters = nn.ModuleList([nn.Sequential(nn.Embedding(num_embeddings, rank), nn.GELU(), nn.Linear(rank, embedding_dim, bias=False)) for _ in range(num_layers)])
    def forward(self, x):
        base_out = self.base(x)
        return sum(base_out[:, i, :] + self.adapters[i](x[:, i, :]) for i in range(len(self.adapters))) / self.num_layers

class SenseVoiceAudioProcessor:
    def __init__(self, frontend): self.frontend = frontend
    def __call__(self, wav, sampling_rate=16000, return_tensors="pt", return_attention_mask=True, **kwargs):
        if isinstance(wav, np.ndarray): wav = torch.from_numpy(wav).float()
        if wav.dim() == 1: wav = wav.unsqueeze(0)
        with torch.no_grad():
            fbank, flen = self.frontend(wav, torch.tensor([wav.size(1)]))
        return SimpleNamespace(input_features=fbank, attention_mask=(torch.arange(fbank.size(1)) < flen[0]).long().unsqueeze(0))


class TalkerModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.talker_config = MiniMindConfig(hidden_size=config.talker_hidden_size, use_moe=config.use_moe)
        self.layers = nn.ModuleList([MiniMindBlock(l, self.talker_config) for l in range(config.num_talker_hidden_layers)])
        self.norm = RMSNorm(config.talker_hidden_size, eps=config.rms_norm_eps)
        self.lm_head = TalkerHead(config.talker_hidden_size, config.audio_vocab_size)
        self.embed_tokens = TalkerEmbedding(config.audio_vocab_size, config.talker_hidden_size)
        self.codec_proj = nn.Sequential(nn.Linear(config.talker_hidden_size, config.talker_hidden_size), nn.GELU(), nn.Linear(config.talker_hidden_size, config.talker_hidden_size), RMSNorm(config.talker_hidden_size, eps=config.rms_norm_eps))
        self.embed_proj = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size), nn.GELU(), nn.Linear(config.hidden_size, config.talker_hidden_size), RMSNorm(config.talker_hidden_size, eps=config.rms_norm_eps))
        self.text_scale, self.audio_scale = nn.Parameter(torch.tensor(3.0)), nn.Parameter(torch.tensor(1.0))
        self.spk_proj = nn.Linear(config.spk_emb_size, config.talker_hidden_size, bias=False)
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=self.talker_config.head_dim, end=config.max_position_embeddings, rope_base=config.rope_theta, rope_scaling=config.rope_scaling)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)


class MiniMindOmni(PreTrainedModel):
    config_class = OmniConfig
    def __init__(self, config: OmniConfig = None, audio_encoder_path="./model/SenseVoiceSmall", vision_model_path="./model/siglip2-base-p32-256-ve", llm_path="./model/Qwen3-0.6B"):
        config = config or OmniConfig()
        super().__init__(config)
        self.config = config
        
        self.qwen = AutoModelForCausalLM.from_pretrained(llm_path, trust_remote_code=True)
        self.thinker = self.qwen.model
        self.lm_head = self.qwen.lm_head
        
        self.config.hidden_size = self.qwen.config.hidden_size
        self.config.num_hidden_layers = self.qwen.config.num_hidden_layers
        self.config.bridge_layer = self.qwen.config.num_hidden_layers // 2 - 1
        
        self.talker = TalkerModule(config)
        self.audio_proj = MMAudioProjector(config.audio_hidden_size, config.hidden_size)
        self.vision_proj = MMVisionProjector(config.image_hidden_size, config.hidden_size, target_tokens=config.image_token_len)
        self.audio_pad_token, self.audio_stop_token, self.audio_spk_token = config.audio_pad_token, config.audio_stop_token, config.audio_spk_token
        audio_encoder, audio_processor = self.load_sensevoice(audio_encoder_path)
        object.__setattr__(self, 'audio_encoder', audio_encoder)
        object.__setattr__(self, 'audio_processor', audio_processor)
        vision_encoder, vision_processor = self.load_vision(vision_model_path)
        object.__setattr__(self, 'vision_encoder', vision_encoder)
        object.__setattr__(self, 'vision_processor', vision_processor)

    @staticmethod
    def load_sensevoice(path):
        if not os.path.exists(path):
            warnings.warn(f"[MiniMindOmni] SenseVoice path not found: {path}")
            return None, None
        logging.getLogger().setLevel(logging.ERROR)
        hf_logging.set_verbosity_error()
        with contextlib.redirect_stdout(io.StringIO()):
            from funasr import AutoModel
            m = AutoModel(model=path, trust_remote_code=True, disable_update=True, device="cpu")
        encoder, frontend = m.model.encoder, m.kwargs["frontend"]
        for p in encoder.parameters(): p.requires_grad = False
        return encoder.eval().float(), SenseVoiceAudioProcessor(frontend.eval())

    @torch.compiler.disable
    def encode_audio_inputs(self, audio_inputs, audio_lens=None):
        if (audio_inputs is None) or (self.audio_encoder is None) or (not audio_inputs.any()): return None
        batch_mask = audio_inputs.flatten(1).any(1)
        enc_dtype = next(self.audio_encoder.parameters()).dtype
        valid_fbank = audio_inputs[batch_mask].to(dtype=enc_dtype)
        if audio_lens is not None:
            valid_lens = audio_lens[batch_mask].to(valid_fbank.device)
        else:
            valid_lens = torch.tensor([valid_fbank.size(1)] * valid_fbank.size(0), device=valid_fbank.device)
        with torch.no_grad():
            emb, _ = self.audio_encoder(valid_fbank, valid_lens)
            
        # [Audio LAST-ViT] 使用频域注意力机制抑制环境噪音/静音帧
        emb = self.apply_last_vit(emb, is_audio=True)
        proj_dtype = next(self.audio_proj.parameters()).dtype
        emb_list = [self.audio_proj(emb[i, :max(1, min(valid_lens[i].item(), emb.size(1)))].unsqueeze(0).to(proj_dtype)).squeeze(0) for i in range(emb.size(0))]
        if batch_mask.all(): return emb_list
        out = [None] * audio_inputs.size(0)
        j = 0
        for i in range(audio_inputs.size(0)):
            if batch_mask[i]:
                out[i] = emb_list[j]
                j += 1
        return out

    @torch.compiler.disable
    def inject_audio_features(self, tokens, h, audio_feats, seqlen):
        if audio_feats is None or not self.config.audio_ids:
            return h
        marker = self.config.audio_ids[0]
        out = []
        for b in range(h.size(0)):
            hb, seq, i = h[b], tokens[b].tolist(), 0
            af = audio_feats[b] if audio_feats[b] is not None else None
            while i < len(seq):
                if seq[i] == marker:
                    start = i
                    while i < len(seq) and seq[i] == marker:
                        i += 1
                    if af is not None:
                        inject_len = min(af.size(0), i - start)
                        hb = torch.cat((hb[:start], af[:inject_len], hb[start + inject_len:]), dim=0)
                        af = None
                else:
                    i += 1
            out.append(hb)
        return torch.stack(out)
    
    @staticmethod
    def load_vision(path):
        if path is None or not os.path.exists(path):
            warnings.warn(f"[MiniMindOmni] Vision model path not found: {path}. vision_encoder will be None!")
            return None, None
        hf_logging.set_verbosity_error()
        try:
            model = SiglipVisionModel.from_pretrained(path)
        except (RuntimeError, ValueError):
            return None, None
        processor = SiglipImageProcessor.from_pretrained(path)
        for p in model.parameters():
            p.requires_grad = False
        return model.eval(), processor

    @torch.compiler.disable
    def apply_last_vit(self, x, is_audio=False):
        """
        LAST-ViT frequency-domain token selection mechanism
        x: [B, N, D]
        """
        orig_dtype = x.dtype
        x = x.float()
        x_detach = x.clone()
        x_fft = torch.fft.fft(x, dim=-1)
        kernel_size = x.shape[-1]
        
        # 区分 audio 和 vision 的 sigma 配置
        if is_audio:
            sigma = getattr(self.config, 'last_vit_audio_sigma', 2.0)
        else:
            sigma = getattr(self.config, 'last_vit_sigma', 2.0)
        
        # 1D Gaussian kernel
        idx = torch.arange(kernel_size, dtype=torch.float32, device=x.device) - kernel_size // 2
        gs_k = torch.exp(-(idx**2) / (2 * sigma**2))
        gs_k = gs_k / gs_k.sum()
        
        x_fft = torch.fft.fftshift(x_fft, dim=-1)
        x_fft = x_fft * gs_k
        x_fft = torch.fft.ifftshift(x_fft, dim=-1)
        x_ifft = torch.fft.ifft(x_fft, dim=-1).real
        
        diff = x_detach / (torch.abs(x_ifft - x_detach) + 1e-6)
        
        # 修正：之前直接对 diff 做 sigmoid，因为 diff 的符号与 x_detach 完全一致且绝对值极大，
        # 会导致 weight 变成 0 或 1，相当于做了一个 ReLU，抹杀了所有负值特征，丢失了 50% 的信息！
        # 现在的做法：计算每个 Patch 在所有通道上的平均 Diff 得分，作为一个全局重要性系数 [B, N, 1]
        patch_score = diff.mean(dim=-1, keepdim=True)
        
        # 将得分在空间维度标准化，避免极值，再通过 sigmoid 映射到 0~1 的平滑权重
        patch_score = (patch_score - patch_score.mean(dim=1, keepdim=True)) / (patch_score.std(dim=1, keepdim=True) + 1e-5)
        weight = torch.sigmoid(patch_score)
        
        # 用平滑权重调节原始特征，这样既抑制了背景，又完美保留了特征向量的方向和正负号！
        sel_p = x_detach * weight
            
        return sel_p.to(orig_dtype)

    @torch.compiler.disable
    def get_image_embeddings(self, image_inputs):
        if hasattr(image_inputs, 'keys'):
            image_inputs = {k: v.squeeze(1) if v.ndim > 2 and v.shape[1] == 1 else v for k, v in image_inputs.items()}
            pixel_attention_mask = image_inputs.get('pixel_attention_mask')
            if pixel_attention_mask is not None and not pixel_attention_mask.any():
                pv = image_inputs['pixel_values']
                return pv.new_zeros(pv.size(0), self.config.image_token_len, self.config.image_hidden_size)
        with torch.no_grad():
            outputs = self.vision_encoder(**image_inputs)
        return self.apply_last_vit(outputs.last_hidden_state, is_audio=False)

    @torch.compiler.disable
    def encode_image_inputs(self, pixel_values):
        if pixel_values is None or self.vision_encoder is None: return None
        mask = pixel_values.flatten(1).any(1)
        if not mask.any(): return pixel_values.new_zeros(pixel_values.size(0), self.config.image_token_len, self.config.hidden_size)
        with torch.no_grad(): emb = self.vision_encoder(pixel_values=pixel_values[mask]).last_hidden_state
        emb = self.apply_last_vit(emb, is_audio=False)
        if emb.dim() == 2: emb = emb.unsqueeze(0)
        emb = self.vision_proj(emb)
        if mask.all(): return emb
        idx = mask.nonzero().view(-1, 1, 1).expand_as(emb)
        return emb.new_zeros(pixel_values.size(0), *emb.shape[1:]).scatter(0, idx, emb)

    @torch.compiler.disable
    def count_vision_proj(self, tokens, h, vision_tensors=None, seqlen=512):
        if vision_tensors is None or not self.config.image_ids:
            return h
        marker, vf = self.config.image_ids[0], vision_tensors
        if vf.dim() == 3:
            vf = vf.unsqueeze(1)
        out = []
        for b in range(h.size(0)):
            hb, seq, k, i = h[b], tokens[b].tolist(), 0, 0
            while i < len(seq):
                if seq[i] == marker:
                    start = i
                    while i < len(seq) and seq[i] == marker:
                        i += 1
                    if k < vf.size(1):
                        hb = torch.cat((hb[:start], vf[b][k][:i - start], hb[i:]), dim=0)[:seqlen]
                        k += 1
                else:
                    i += 1
            out.append(hb)
        return torch.stack(out)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, logits_to_keep=0, audio_inputs=None, audio_lens=None, pixel_values=None, **args):
        if len(input_ids.shape) == 2:
            batch_size, seq_length = input_ids.shape
            text_ids = input_ids
            audio_ids = torch.full((batch_size, 8, seq_length), self.audio_pad_token, dtype=torch.long, device=input_ids.device)
        else:
            batch_size, _, seq_length = input_ids.shape
            text_ids, audio_ids = input_ids[:, 8, :], input_ids[:, :8, :]
        
        n_talker = len(self.talker.layers)
        
        if past_key_values is None:
            thinker_kv = None
            talker_kvs = [None] * n_talker
            start_pos = 0
        else:
            thinker_kv = past_key_values[0]
            talker_kvs = past_key_values[1:]
            start_pos = talker_kvs[0][0].shape[1] if talker_kvs[0] is not None else 0

        if self.talker.freqs_cos[0, 0] == 0:
            freqs_cos, freqs_sin = precompute_freqs_cis(dim=self.talker.talker_config.head_dim, end=self.config.max_position_embeddings, rope_base=self.config.rope_theta, rope_scaling=self.config.rope_scaling)
            self.talker.freqs_cos, self.talker.freqs_sin = freqs_cos.to(input_ids.device), freqs_sin.to(input_ids.device)

        # ======= Thinker: text-only input, output text logits =======
        inputs_embeds = self.thinker.embed_tokens(text_ids)
        if audio_inputs is not None and start_pos == 0:
            audio_features = self.encode_audio_inputs(audio_inputs, audio_lens)
            inputs_embeds = self.inject_audio_features(text_ids, inputs_embeds, audio_features, seq_length)
        if pixel_values is not None and start_pos == 0:
            if hasattr(pixel_values, 'keys'):
                img_emb = self.get_image_embeddings(pixel_values).to(inputs_embeds.dtype)
                vision_tensors = self.vision_proj(img_emb)
            else:
                if len(pixel_values.shape) == 6:
                    pixel_values = pixel_values.squeeze(2)
                if len(pixel_values.shape) == 4:
                    pixel_values = pixel_values.unsqueeze(1)
                bs, num, c, im_h, im_w = pixel_values.shape
                stack_dim = 1 if bs > 1 else 0
                vision_tensors = torch.stack([
                    self.encode_image_inputs(pixel_values[:, i, :, :, :])
                    for i in range(num)
                ], dim=stack_dim)
            inputs_embeds = self.count_vision_proj(tokens=text_ids, h=inputs_embeds, vision_tensors=vision_tensors, seqlen=seq_length)
        
        outputs = self.thinker(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=thinker_kv,
            use_cache=use_cache,
            output_hidden_states=True
        )
        h_thinker = outputs.last_hidden_state
        bridge_states = outputs.hidden_states[self.config.bridge_layer + 1]
        presents = [outputs.past_key_values] if outputs.past_key_values is not None else []

        # ======= Talker: thinker hidden + audio codes, output audio logits =======
        talker_emb = self.talker.embed_tokens(audio_ids)
        spk_emb = args.get('spk_emb', None)
        if spk_emb is not None:
            spk_mask = (audio_ids[:, 0, :] == self.audio_spk_token).unsqueeze(-1)
            talker_emb = torch.where(spk_mask, self.talker.spk_proj(spk_emb).unsqueeze(1), talker_emb)
        hidden_states = self.talker.embed_proj(bridge_states) * self.talker.text_scale + self.talker.codec_proj(talker_emb) * self.talker.audio_scale
        talker_pos_emb = (self.talker.freqs_cos[start_pos:start_pos + seq_length], self.talker.freqs_sin[start_pos:start_pos + seq_length])
        for layer, past_key_value in zip(self.talker.layers, talker_kvs):
            hidden_states, present = layer(hidden_states, talker_pos_emb, past_key_value=past_key_value, use_cache=use_cache, attention_mask=attention_mask)
            if use_cache:
                presents.append(present)
        h_talker = self.talker.norm(hidden_states)

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        aux_loss = sum(l.mlp.aux_loss for l in list(self.talker.layers) if isinstance(l.mlp, MOEFeedForward))
        aux_loss += sum(p.sum() for p in self.audio_proj.parameters()) * 0 + sum(p.sum() for p in self.vision_proj.parameters()) * 0 + sum(p.sum() for p in self.talker.lm_head.adapters.parameters()) * 0 + sum(p.sum() for p in self.talker.spk_proj.parameters()) * 0 # dummy gradient
        text_logits = self.lm_head(h_thinker[:, slice_indices, :])
        audio_logits = self.talker.lm_head(h_talker[:, slice_indices, :])
        
        out = MoeCausalLMOutputWithPast(aux_loss=aux_loss, logits=text_logits, past_key_values=presents if use_cache else None)
        out.audio_logits = audio_logits
        return out

    @torch.inference_mode()
    def generate(self, input_ids, eos_token_id=151645, max_new_tokens=1024, temperature=0.75, top_p=0.90,
                 stream=False, rp=1., use_cache=True, return_audio_codes=False, **args):
        if stream:
            return self.stream_generate(input_ids, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache, return_audio_codes, **args)
        tokens = list(self.stream_generate(input_ids, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache, return_audio_codes, **args))
        return tokens[-1] if tokens else input_ids

    def stream_generate(self, input_ids, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache, return_audio_codes=False, **args):
        start_pos, past_kvs, text_finished, first_finished = input_ids.shape[1], None, False, True
        audio_codes = [[] for _ in range(8)]
        audio_stop_pos = [None] * 8
        audio_buffer = torch.full((1, 8, start_pos), self.audio_pad_token, dtype=torch.long, device=input_ids.device)
        spk_emb = args.get('spk_emb', None)
        ref_codes = args.get('ref_codes', None)
        ref_len = ref_codes.shape[2] if ref_codes is not None else 0
        spk_reserve = 1 if spk_emb is not None else 0
        fill_end = start_pos
        fill_start = max(spk_reserve, start_pos - ref_len)
        if ref_codes is not None and fill_start < fill_end:
            audio_buffer[:, :, fill_start:fill_end] = ref_codes[:, :, -(fill_end - fill_start):]
        if spk_emb is not None and fill_start > 0:
            audio_buffer[:, :, fill_start - 1] = self.audio_spk_token
        think_end_step, generated_tokens = None, ([] if args.get('open_thinking', False) else None)
        while input_ids.shape[1] < start_pos + max_new_tokens:
            if past_kvs is None or not use_cache:
                out = self.forward(torch.cat((audio_buffer, input_ids.unsqueeze(1)), dim=1), past_key_values=past_kvs, use_cache=use_cache, **args)
            else:
                out = self.forward(torch.cat((audio_buffer[:, :, -1:], input_ids[:, -1:].unsqueeze(1)), dim=1), past_key_values=past_kvs, use_cache=use_cache, **args)
            past_kvs = out.past_key_values

            logits = out.logits[0, -1, :].clone() / (temperature + 1e-9)
            if rp != 1.0:
                seen = list(set(input_ids[0].tolist())); score = logits[seen]; logits[seen] = torch.where(score > 0, score / rp, score * rp)
            if top_p and top_p < 1.0:
                sorted_l, sorted_i = torch.sort(logits, descending=True)
                mask = torch.cumsum(F.softmax(sorted_l, dim=-1), dim=-1) > top_p
                mask[1:], mask[0] = mask[:-1].clone(), False
                logits[sorted_i[mask]] = -float('Inf')
            text_token = torch.multinomial(F.softmax(logits, dim=-1), 1).item()

            if text_finished:
                text_token = args.get('enter_token_id', 201) if first_finished else args.get('pad_token_id', 0)
                first_finished = False

            step = input_ids.shape[1] - start_pos  # 已生成token数（0=首次，此时模型处理prompt末尾token）
            audio_step = step - 1  # 延迟1步：输出第1个text时无audio，输出第2个text时layer0开始
            if generated_tokens is not None:
                generated_tokens.append(text_token)
                if not think_end_step and generated_tokens[-len(self.config.think_end_ids):] == list(self.config.think_end_ids): think_end_step = step + 2
                audio_step = (step - think_end_step) if think_end_step else -1
            for i, al in enumerate(out.audio_logits):
                if audio_step < i:
                    audio_codes[i].append(self.audio_pad_token)
                else:
                    logits_i = al[0, -1, :].clone() / 0.2
                    for prev_code in audio_codes[i][-3:]: score = logits_i[prev_code]; logits_i[prev_code] = torch.where(score > 0, score / 1.05, score * 1.05)
                    top_val, top_idx = logits_i.topk(50)
                    code = top_idx[torch.multinomial(F.softmax(top_val, dim=-1), 1)].item()
                    audio_codes[i].append(code)
                    if audio_stop_pos[i] is None and code >= 2048: audio_stop_pos[i] = len(audio_codes[i]) - 1

            if text_finished and all(audio_stop_pos[i] is not None for i in range(8)): break

            input_ids = torch.cat((input_ids, torch.tensor([[text_token]], device=input_ids.device)), dim=1)
            audio_buffer = torch.cat((audio_buffer, torch.full((1, 8, 1), self.audio_pad_token, dtype=torch.long, device=input_ids.device)), dim=2)
            for i in range(min(audio_step + 1, 8)): audio_buffer[0, i, -1] = audio_codes[i][-1]

            audio_frame = None
            if return_audio_codes and audio_step >= 7:
                frame = [audio_codes[i][step - 7 + i] for i in range(8)]
                active_layers = sum(1 for i in range(8) if audio_stop_pos[i] is None or step - 7 + i < audio_stop_pos[i])
                if active_layers >= 8: audio_frame = frame
            if not text_finished:
                yield input_ids[:, start_pos:], audio_frame
                if text_token == eos_token_id: text_finished = True
            else:
                yield None, audio_frame


# ==== Realtime VAD (与模型本体零耦合，纯工程层) ====
class SileroVAD:
    def __init__(self, path):
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = opts.intra_op_num_threads = 1
        opts.log_severity_level = 4
        self.session = ort.InferenceSession(path, providers=["CPUExecutionProvider"], sess_options=opts)
        self.h, self.c = np.zeros((2, 1, 64), dtype=np.float32), np.zeros((2, 1, 64), dtype=np.float32)

    def reset(self):
        self.h[:], self.c[:] = 0, 0

    def __call__(self, chunk, sr=16000):
        out, self.h, self.c = self.session.run(None, {"input": chunk.reshape(1, -1).astype(np.float32), "h": self.h, "c": self.c, "sr": np.array(sr, dtype="int64")})
        return float(out[0][0])


class RealtimeSession:
    def __init__(self, vad_path, sr=16000, threshold=0.8, min_speech_ms=128, min_silence_ms=800):
        self.vad, self.sr, self.threshold = SileroVAD(vad_path), sr, threshold
        self.min_speech, self.min_silence = int(sr * min_speech_ms / 1000), int(sr * min_silence_ms / 1000)
        self.reset()

    def reset(self):
        self.vad.reset()
        self.buffer, self.ring, self.speaking, self.generating, self.interrupt = [], [], False, False, False
        self.speech_samples = self.silence_samples = self.tail_silence = 0

    def push_chunk(self, chunk, W=1024):
        for i in range(0, max(len(chunk), 1), W):
            w = chunk[i:i + W]
            if len(w) < W:
                w = np.pad(w, (0, W - len(w)))
            prob = self.vad(w, self.sr)
            if prob > self.threshold:
                self.silence_samples = self.tail_silence = 0
                self.speech_samples += len(w)
                self.buffer.append(w)
                if self.speech_samples >= self.min_speech and not self.speaking:
                    self.speaking = True
                    self.buffer = self.ring + self.buffer
                    self.ring = []
                if self.generating and self.speaking:
                    self.interrupt = True
                    return 'interrupt'
            elif self.speaking:
                self.silence_samples += len(w)
                self.tail_silence += 1
                self.buffer.append(w)
                if self.silence_samples >= self.min_silence:
                    if self.tail_silence > 1:
                        del self.buffer[-(self.tail_silence - 1):]
                    self.speaking, self.speech_samples, self.silence_samples, self.tail_silence = False, 0, 0, 0
                    return 'speech_end'
            else:
                if self.speech_samples > 0:
                    self.buffer.clear()
                self.speech_samples = 0
                self.ring = [w]
        return 'listening'

    def get_audio(self):
        audio = np.concatenate(self.buffer) if self.buffer else np.array([], dtype=np.float32)
        self.buffer.clear()
        return audio