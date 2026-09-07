"""Quality-aware DiT motion prior for world-space dual-hand inference."""

from typing import Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimestepEmbedding(nn.Module):
    """Sinusoidal positional embedding for timesteps."""

    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / half
        )
        args = t[:, None].float() * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class ContinuousTimestepEmbedding(nn.Module):
    """Embed continuous timesteps t in [0, 1] for flow matching."""

    def __init__(self, embedding_dim: int, freq_dim: int = 256):
        super().__init__()
        self.sinusoidal = SinusoidalTimestepEmbedding(freq_dim)
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_scaled = t * 1000.0
        emb = self.sinusoidal(t_scaled)
        return self.mlp(emb)


def get_1d_rotary_pos_embed(dim: int, seq_len: int, device: torch.device = None,
                            offset: int = 0):
    """Compute 1D rotary positional embeddings (cos, sin)."""
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(offset, offset + seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    cos_emb = freqs.cos()
    sin_emb = freqs.sin()
    return cos_emb, sin_emb


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Apply rotary positional embedding to input tensor."""
    d = cos.shape[-1]
    x1 = x[..., :d]
    x2 = x[..., d:2*d]
    x_rest = x[..., 2*d:]
    out1 = x1 * cos - x2 * sin
    out2 = x2 * cos + x1 * sin
    return torch.cat([out1, out2, x_rest], dim=-1)


class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional rotary embeddings and cross-attention."""

    def __init__(self, dim: int, num_heads: int, head_dim: int, dropout: float = 0.0,
                 cross_attention_dim: Optional[int] = None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        self.scale = head_dim ** -0.5

        kv_dim = cross_attention_dim or dim
        self.to_q = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_k = nn.Linear(kv_dim, self.inner_dim, bias=False)
        self.to_v = nn.Linear(kv_dim, self.inner_dim, bias=False)
        self.to_out = nn.Linear(self.inner_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Attend over x or encoder tokens, with optional rotary embeddings and masks."""
        B, T, _ = x.shape
        kv_input = encoder_hidden_states if encoder_hidden_states is not None else x

        q = self.to_q(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.to_k(kv_input).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(kv_input).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        S = k.shape[2]

        if rotary_emb is not None and encoder_hidden_states is None:
            cos, sin = rotary_emb
            cos = cos[:T].unsqueeze(0).unsqueeze(0)
            sin = sin[:T].unsqueeze(0).unsqueeze(0)
            q = apply_rotary_pos_emb(q, cos, sin)
            k = apply_rotary_pos_emb(k, cos, sin)

        mask = None
        if attention_mask is not None and encoder_hidden_states is None:
            if attention_mask.dim() == 3:
                mask = attention_mask[:, None, :, :].expand(-1, self.num_heads, -1, -1)
            else:
                mask = attention_mask[:, None, None, :].expand(-1, self.num_heads, T, -1)
            mask = mask.bool()

        if attn_bias is not None:
            bias = attn_bias[:, None, None, :]
            if mask is not None:
                mask = mask.float() + bias
            else:
                mask = bias.expand(-1, self.num_heads, T, -1)

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=self.dropout.p if self.training else 0.0,
        )

        out = out.transpose(1, 2).reshape(B, T, self.inner_dim)
        return self.to_out(out)


class SwiGLU(nn.Module):
    """SwiGLU activation for feed-forward network."""

    def __init__(self, dim: int, inner_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        inner_dim = inner_dim or dim * 4
        self.w1 = nn.Linear(dim, inner_dim, bias=True)
        self.w2 = nn.Linear(dim, inner_dim, bias=True)
        self.w3 = nn.Linear(inner_dim, dim, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))


class HeadMotionEncoder(nn.Module):
    """Encode per-frame head motion into conditioning tokens."""

    def __init__(self, head_feat_dim: int = 9, d_model: int = 512):
        super().__init__()
        self.per_frame_proj = nn.Sequential(
            nn.Linear(head_feat_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.global_pool = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, head_feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        per_frame = self.per_frame_proj(head_feats)
        global_emb = self.global_pool(per_frame.mean(dim=1))
        return per_frame, global_emb


class HandDiTBlock(nn.Module):
    """DiT block with self-attention, ordered conditioning cross-attention and SwiGLU."""

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout: float = 0.0,
        norm_eps: float = 1e-5,
        ff_inner_dim: Optional[int] = None,
        has_scene_cross_attn: bool = False,
        has_action_cross_attn: bool = False,
        cross_attn_order: Tuple[str, ...] = ("scene", "action", "head", "obs"),
    ):
        super().__init__()
        if has_action_cross_attn:
            raise ValueError("Released motion priors do not use action cross-attention")
        self.has_scene_cross_attn = has_scene_cross_attn
        # Checkpoints retain an inactive action entry in their conditioning order.
        allowed = {"scene", "action", "head", "obs"}
        order = list(cross_attn_order)
        assert set(order) == allowed, f"cross_attn_order must be a permutation of {allowed}, got {order}"
        self.cross_attn_order: Tuple[str, ...] = tuple(
            s for s in order
            if (s == "scene" and has_scene_cross_attn)
            or s in ("head", "obs")
        )

        self.norm1 = nn.LayerNorm(dim, eps=norm_eps, elementwise_affine=False)
        self.attn1 = MultiHeadAttention(dim, num_attention_heads, attention_head_dim, dropout)

        if has_scene_cross_attn:
            self.norm_scene = nn.LayerNorm(dim, eps=norm_eps, elementwise_affine=False)
            self.cross_attn_scene = MultiHeadAttention(
                dim, num_attention_heads, attention_head_dim, dropout, cross_attention_dim=dim,
            )


        self.norm_head = nn.LayerNorm(dim, eps=norm_eps, elementwise_affine=False)
        self.cross_attn_head = MultiHeadAttention(
            dim, num_attention_heads, attention_head_dim, dropout, cross_attention_dim=dim,
        )

        self.norm_obs = nn.LayerNorm(dim, eps=norm_eps, elementwise_affine=False)
        self.cross_attn_obs = MultiHeadAttention(
            dim, num_attention_heads, attention_head_dim, dropout, cross_attention_dim=dim,
        )

        self.norm2 = nn.LayerNorm(dim, eps=norm_eps, elementwise_affine=False)
        self.ff = SwiGLU(dim, inner_dim=ff_inner_dim, dropout=dropout)

        n_mod = 6 + 3 * len(self.cross_attn_order)
        self.scale_shift_table = nn.Parameter(torch.randn(1, 1, n_mod, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        scene_tokens: Optional[torch.Tensor] = None,
        head_motion_tokens: Optional[torch.Tensor] = None,
        obs_gated_tokens: Optional[torch.Tensor] = None,
        obs_attn_bias: Optional[torch.Tensor] = None,
        rotary_embedding: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        time_hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T = hidden_states.shape[:2]

        n_mod = 6 + 3 * len(self.cross_attn_order)
        chunks = (
            self.scale_shift_table + time_hidden_states.reshape(B, T, n_mod, -1)
        ).chunk(n_mod, dim=-2)
        params = [c.squeeze(-2) for c in chunks]

        shift_sa, scale_sa, gate_sa = params[0:3]
        h = self.norm1(hidden_states)
        h = h * (1 + scale_sa) + shift_sa
        h = self.attn1(h, attention_mask=attention_mask, rotary_emb=rotary_embedding)
        hidden_states = gate_sa * h + hidden_states

        idx = 3
        for stage in self.cross_attn_order:
            shift, scale, gate = params[idx:idx + 3]
            idx += 3
            if stage == "scene":
                if scene_tokens is None:
                    continue
                h = self.norm_scene(hidden_states)
                h = h * (1 + scale) + shift
                h = self.cross_attn_scene(h, encoder_hidden_states=scene_tokens)
                hidden_states = gate * h + hidden_states
            elif stage == "head":
                if head_motion_tokens is None:
                    continue
                h = self.norm_head(hidden_states)
                h = h * (1 + scale) + shift
                h = self.cross_attn_head(h, encoder_hidden_states=head_motion_tokens)
                hidden_states = gate * h + hidden_states
            elif stage == "obs":
                if obs_gated_tokens is None:
                    continue
                h = self.norm_obs(hidden_states)
                h = h * (1 + scale) + shift
                h = self.cross_attn_obs(h, encoder_hidden_states=obs_gated_tokens,
                                        attn_bias=obs_attn_bias)
                hidden_states = gate * h + hidden_states

        shift_ff, scale_ff, gate_ff = params[idx:idx + 3]
        h = self.norm2(hidden_states)
        h = h * (1 + scale_ff) + shift_ff
        h = self.ff(h)
        hidden_states = gate_ff * h + hidden_states

        return hidden_states


class HandDiTModel(nn.Module):
    """Flow-Matching DiT for hand motion generation with quality-gated conditioning."""

    def __init__(
        self,
        hand_feat_dim: int = 48,
        head_feat_dim: int = 9,
        betas_dim: int = 20,
        num_layers: int = 8,
        attention_head_dim: int = 64,
        num_attention_heads: int = 8,
        zero_init: bool = False,
        disable_crop_embed: bool = False,
        disable_tarope: bool = False,
        disable_eff_t: bool = False,
        scene_feat_dim: int = 0,
        action_feat_dim_dit: int = 0,
        modality_tokens: bool = False,
        q_adaln_dim: int = 0,
        q_adaln_perhand: bool = False,
        q_granularity: int = 2,
        depth_cond_dim: int = 0,
        cross_attn_order: Tuple[str, ...] = ("scene", "action", "head", "obs"),
        detection_aware_attn: bool = False,
        detection_attn_q_thresh: float = 0.05,
        betas_input_ln: bool = False,
        presence_head: bool = False,
        presence_head_input: str = "hidden",
    ):
        super().__init__()
        unsupported = {
            "action_feat_dim_dit": action_feat_dim_dit,
            "q_adaln_perhand": q_adaln_perhand,
            "depth_cond_dim": depth_cond_dim,
            "detection_aware_attn": detection_aware_attn,
            "presence_head": presence_head,
        }
        enabled = [name for name, value in unsupported.items() if value]
        if enabled:
            raise ValueError("Unsupported motion-prior options: " + ", ".join(enabled))
        self.cross_attn_order = tuple(cross_attn_order)

        self.hand_feat_dim = hand_feat_dim
        self.inner_dim = num_attention_heads * attention_head_dim
        self.rotary_embed_dim = attention_head_dim // 2
        self.disable_crop_embed = disable_crop_embed
        self.disable_tarope = disable_tarope
        self.disable_eff_t = disable_eff_t
        self.scene_feat_dim = scene_feat_dim
        self.modality_tokens = modality_tokens
        self.q_adaln_dim = q_adaln_dim
        if q_granularity != 2:
            raise ValueError("Released motion priors use q_granularity=2")
        self.q_granularity = q_granularity

        self.time_embed = ContinuousTimestepEmbedding(self.inner_dim)

        _betas_layers = [nn.LayerNorm(betas_dim)] if betas_input_ln else []
        _betas_layers += [
            nn.Linear(betas_dim, self.inner_dim),
            nn.GELU(),
            nn.Linear(self.inner_dim, self.inner_dim),
        ]
        self.betas_embed = nn.Sequential(*_betas_layers)

        self.head_encoder = HeadMotionEncoder(head_feat_dim, self.inner_dim)

        self.scene_encoder = None
        if scene_feat_dim > 0:
            self.scene_encoder = nn.Sequential(
                nn.LayerNorm(scene_feat_dim),
                nn.Linear(scene_feat_dim, self.inner_dim),
                nn.GELU(),
                nn.Linear(self.inner_dim, self.inner_dim),
            )


        self.crop_embed = nn.Sequential(
            SinusoidalTimestepEmbedding(64),
        )
        self.crop_proj = nn.Sequential(
            nn.Linear(64 * 2, self.inner_dim),
            nn.GELU(),
            nn.Linear(self.inner_dim, self.inner_dim),
        )

        has_scene = scene_feat_dim > 0
        n_mod = 12 + (3 if has_scene else 0)
        self.adaln_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.inner_dim, n_mod * self.inner_dim),
        )

        self.q_embed = None
        if q_adaln_dim > 0:
            q_in_dim = q_adaln_dim
            self.q_embed = nn.Sequential(
                nn.Linear(q_in_dim, self.inner_dim),
                nn.SiLU(),
                nn.Linear(self.inner_dim, self.inner_dim),
            )
            nn.init.zeros_(self.q_embed[-1].weight)
            nn.init.zeros_(self.q_embed[-1].bias)


        proj_in_dim = hand_feat_dim + (0 if disable_eff_t else 2)
        self.proj_in = nn.Linear(proj_in_dim, self.inner_dim, bias=False)
        self.preprocess_conv = nn.Conv1d(hand_feat_dim, hand_feat_dim, 1, bias=False)
        self.proj_out = nn.Linear(self.inner_dim, hand_feat_dim, bias=False)
        self.postprocess_conv = nn.Conv1d(hand_feat_dim, hand_feat_dim, 1, bias=False)

        if modality_tokens:
            eff_t_extra = 0 if disable_eff_t else 1
            self.wrist_proj_in = nn.Linear(9 + eff_t_extra, self.inner_dim, bias=False)
            self.fingers_proj_in = nn.Linear(45 + eff_t_extra, self.inner_dim, bias=False)
            self.modal_embed = nn.Parameter(torch.zeros(4, self.inner_dim))
            nn.init.normal_(self.modal_embed, std=0.02)
            self.wrist_proj_out = nn.Linear(self.inner_dim, 9, bias=False)
            self.fingers_proj_out = nn.Linear(self.inner_dim, 45, bias=False)

        self.transformer_blocks = nn.ModuleList([
            HandDiTBlock(
                dim=self.inner_dim,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                has_scene_cross_attn=has_scene,
                cross_attn_order=self.cross_attn_order,
            )
            for _ in range(num_layers)
        ])

        if zero_init:
            self._zero_init_output()

    def _zero_init_output(self):
        for p in self.proj_out.parameters():
            nn.init.zeros_(p)
        for p in self.postprocess_conv.parameters():
            nn.init.zeros_(p)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        timestep: torch.Tensor,
        head_motion_feats: Optional[torch.FloatTensor] = None,
        obs_gated_tokens: Optional[torch.FloatTensor] = None,
        obs_attn_bias: Optional[torch.FloatTensor] = None,
        scene_feats: Optional[torch.FloatTensor] = None,
        betas: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        crop_info: Optional[torch.FloatTensor] = None,
        effective_t: Optional[torch.FloatTensor] = None,
        q_per_frame: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> torch.FloatTensor:
        """Forward pass."""

        if self.modality_tokens:
            return self._forward_modality_tokens(
                hidden_states, timestep, head_motion_feats, obs_gated_tokens,
                obs_attn_bias, scene_feats, betas, attention_mask, crop_info,
                effective_t, q_per_frame=q_per_frame,
                **kwargs)

        hidden_states = self.preprocess_conv(hidden_states) + hidden_states

        hidden_states = hidden_states.transpose(1, 2)

        expects_eff_t = self.proj_in.in_features > self.hand_feat_dim
        if expects_eff_t:
            if effective_t is not None:
                eff_t = effective_t.transpose(1, 2)
            else:
                B_size = hidden_states.shape[0]
                T_size = hidden_states.shape[1]
                t_val = timestep[:, None, None].expand(B_size, T_size, 1)
                eff_t = t_val.expand(-1, -1, 2)
            hidden_states = self.proj_in(torch.cat([hidden_states, eff_t], dim=-1))
        else:
            hidden_states = self.proj_in(hidden_states)

        B, T, _ = hidden_states.shape

        t_emb = self.time_embed(timestep)

        head_tokens = None
        head_global = torch.zeros_like(t_emb)
        if head_motion_feats is not None:
            head_tokens, head_global = self.head_encoder(head_motion_feats)

        betas_emb = torch.zeros_like(t_emb)
        if betas is not None:
            betas_emb = self.betas_embed(betas)

        crop_emb = torch.zeros_like(t_emb)
        abs_offset = 0
        if crop_info is not None:
            if not self.disable_crop_embed:
                start_frac = crop_info[:, 0] * 1000.0
                len_frac = crop_info[:, 1] * 1000.0
                emb_start = self.crop_embed(start_frac)
                emb_len = self.crop_embed(len_frac)
                crop_emb = self.crop_proj(torch.cat([emb_start, emb_len], dim=-1))
            if not self.disable_tarope:
                abs_offset = int(crop_info[0, 2].item())

        cond = t_emb + head_global + betas_emb + crop_emb
        _has_q = self.q_embed is not None and q_per_frame is not None
        if _has_q:
            cond_per_frame = cond.unsqueeze(1) + self.q_embed(q_per_frame)
            time_hidden_states = self.adaln_proj(cond_per_frame)
        else:
            time_hidden_states = self.adaln_proj(cond)
            time_hidden_states = time_hidden_states.unsqueeze(1).expand(-1, T, -1)

        scene_tokens = None
        if self.scene_encoder is not None and scene_feats is not None:
            scene_tokens = self.scene_encoder(scene_feats)
            if scene_tokens.dim() == 4:
                B_s, T_s, N_s, D_s = scene_tokens.shape
                scene_tokens = scene_tokens.reshape(B_s, T_s * N_s, D_s)


        rotary_embedding = get_1d_rotary_pos_embed(
            self.rotary_embed_dim, T, device=hidden_states.device,
            offset=abs_offset,
        )

        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                scene_tokens=scene_tokens,
                head_motion_tokens=head_tokens,
                obs_gated_tokens=obs_gated_tokens,
                obs_attn_bias=obs_attn_bias,
                rotary_embedding=rotary_embedding,
                time_hidden_states=time_hidden_states,
            )


        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.postprocess_conv(hidden_states) + hidden_states

        return hidden_states

    def _forward_modality_tokens(
        self,
        hidden_states: torch.FloatTensor,
        timestep: torch.Tensor,
        head_motion_feats=None, obs_gated_tokens=None, obs_attn_bias=None,
        scene_feats=None, betas=None, attention_mask=None, crop_info=None,
        effective_t=None, q_per_frame=None,
        **kwargs,
    ) -> torch.FloatTensor:
        """Forward with modality-specific tokenization (4 tokens per frame)."""
        B, C, T = hidden_states.shape

        x = hidden_states.transpose(1, 2)
        wrist_L = torch.cat([x[:, :, 0:6], x[:, :, 51:54]], dim=-1)
        fingers_L = x[:, :, 6:51]
        wrist_R = torch.cat([x[:, :, 54:60], x[:, :, 105:108]], dim=-1)
        fingers_R = x[:, :, 60:105]

        use_eff_t = not self.disable_eff_t
        if use_eff_t:
            if effective_t is not None:
                n_eff = effective_t.shape[1]
                if n_eff == 4:
                    eff_wL = effective_t[:, 0:1, :].transpose(1, 2)
                    eff_fL = effective_t[:, 1:2, :].transpose(1, 2)
                    eff_wR = effective_t[:, 2:3, :].transpose(1, 2)
                    eff_fR = effective_t[:, 3:4, :].transpose(1, 2)
                else:
                    eff_wL = eff_fL = effective_t[:, 0:1, :].transpose(1, 2)
                    eff_wR = eff_fR = effective_t[:, 1:2, :].transpose(1, 2)
            else:
                t_val = timestep[:, None, None].expand(B, T, 1)
                eff_wL = eff_fL = eff_wR = eff_fR = t_val
            wrist_L = torch.cat([wrist_L, eff_wL], dim=-1)
            fingers_L = torch.cat([fingers_L, eff_fL], dim=-1)
            wrist_R = torch.cat([wrist_R, eff_wR], dim=-1)
            fingers_R = torch.cat([fingers_R, eff_fR], dim=-1)

        tok_wL = self.wrist_proj_in(wrist_L) + self.modal_embed[0]
        tok_fL = self.fingers_proj_in(fingers_L) + self.modal_embed[1]
        tok_wR = self.wrist_proj_in(wrist_R) + self.modal_embed[2]
        tok_fR = self.fingers_proj_in(fingers_R) + self.modal_embed[3]

        stacked = torch.stack([tok_wL, tok_fL, tok_wR, tok_fR], dim=2)
        hidden_states = stacked.reshape(B, T * 4, self.inner_dim)
        S = T * 4

        if attention_mask is not None:
            attention_mask = attention_mask.repeat_interleave(4, dim=1)


        t_emb = self.time_embed(timestep)
        head_tokens = None
        head_global = torch.zeros_like(t_emb)
        if head_motion_feats is not None:
            head_tokens, head_global = self.head_encoder(head_motion_feats)

        betas_emb = torch.zeros_like(t_emb)
        if betas is not None:
            betas_emb = self.betas_embed(betas)

        crop_emb = torch.zeros_like(t_emb)
        abs_offset = 0
        if crop_info is not None:
            if not self.disable_crop_embed:
                start_frac = crop_info[:, 0] * 1000.0
                len_frac = crop_info[:, 1] * 1000.0
                emb_start = self.crop_embed(start_frac)
                emb_len = self.crop_embed(len_frac)
                crop_emb = self.crop_proj(torch.cat([emb_start, emb_len], dim=-1))
            if not self.disable_tarope:
                abs_offset = int(crop_info[0, 2].item())

        cond = t_emb + head_global + betas_emb + crop_emb
        _has_q = self.q_embed is not None and q_per_frame is not None
        if _has_q:
            q_emb = self.q_embed(q_per_frame)
            cond_per_token = cond.unsqueeze(1) + q_emb.repeat_interleave(4, dim=1)
            time_hidden_states = self.adaln_proj(cond_per_token)
        else:
            time_hidden_states = self.adaln_proj(cond)
            time_hidden_states = time_hidden_states.unsqueeze(1).expand(-1, S, -1)

        scene_tokens = None
        if self.scene_encoder is not None and scene_feats is not None:
            scene_tokens = self.scene_encoder(scene_feats)


        rotary_embedding = get_1d_rotary_pos_embed(
            self.rotary_embed_dim, T, device=hidden_states.device,
            offset=abs_offset)
        cos, sin = rotary_embedding
        cos = cos.repeat_interleave(4, dim=0)
        sin = sin.repeat_interleave(4, dim=0)
        rotary_embedding = (cos, sin)

        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                scene_tokens=scene_tokens,
                head_motion_tokens=head_tokens,
                obs_gated_tokens=obs_gated_tokens,
                obs_attn_bias=obs_attn_bias,
                rotary_embedding=rotary_embedding,
                time_hidden_states=time_hidden_states)

        hidden_states = hidden_states.reshape(B, T, 4, self.inner_dim)
        out_wL = self.wrist_proj_out(hidden_states[:, :, 0])
        out_fL = self.fingers_proj_out(hidden_states[:, :, 1])
        out_wR = self.wrist_proj_out(hidden_states[:, :, 2])
        out_fR = self.fingers_proj_out(hidden_states[:, :, 3])

        output = torch.cat([
            out_wL[:, :, :6], out_fL, out_wL[:, :, 6:9],
            out_wR[:, :, :6], out_fR, out_wR[:, :, 6:9],
        ], dim=-1)

        return output.transpose(1, 2)
