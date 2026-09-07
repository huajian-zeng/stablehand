"""Quality estimation and per-hand visual conditioning for released checkpoints."""

import torch
import torch.nn as nn
from torch import Tensor

SINGLE_HAND_DIM = 54


class QualityNetworkTransformer(nn.Module):
    """Estimate each hand's proposal quality from motion, camera and depth cues."""

    def __init__(
        self,
        hand_dim: int = SINGLE_HAND_DIM,
        visual_dim: int = 0,
        visual_perhand: bool = False,
        visual_proj_dim: int = 0,
        cam_feat_dim: int = 0,
        kpt2d_dim: int = 0,
        delta_dim: int = 0,
        da3_depth_dim: int = 0,
        da3_depth_layernorm: bool = False,
        mask_dim: int = 0,
        depth_drop_abs: bool = False,
        use_geo_token: bool = False,
        geo_layernorm: bool = False,
        proposal_trans_layernorm: bool = False,
        visual_layernorm: bool = False,
        cross_hand_attention: bool = False,
        output_percomp: bool = False,
        q_granularity: int = 2,
        scene_feat_dim: int = 0,
        cam_ca_dim: int = 0,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        output_mode: str = "q",
        use_det_flag: bool = True,
    ):
        super().__init__()
        unsupported = {
            "visual_perhand": visual_perhand,
            "visual_proj_dim": visual_proj_dim,
            "kpt2d_dim": kpt2d_dim,
            "delta_dim": delta_dim,
            "da3_depth_layernorm": da3_depth_layernorm,
            "mask_dim": mask_dim,
            "depth_drop_abs": depth_drop_abs,
            "geo_layernorm": geo_layernorm,
            "proposal_trans_layernorm": proposal_trans_layernorm,
            "visual_layernorm": visual_layernorm,
            "cross_hand_attention": cross_hand_attention,
            "scene_feat_dim": scene_feat_dim,
            "cam_ca_dim": cam_ca_dim,
        }
        enabled = [name for name, value in unsupported.items() if value]
        if enabled:
            raise ValueError("Unsupported Quality Network options: " + ", ".join(enabled))
        if q_granularity != 2:
            raise ValueError("Released Quality Networks use q_granularity=2")
        if output_mode not in ("q", "error"):
            raise ValueError("output_mode must be 'q' or 'error'")

        self.hand_dim = hand_dim
        self.visual_dim = visual_dim
        self.cam_feat_dim = cam_feat_dim
        self.da3_depth_dim = da3_depth_dim
        self.use_geo_token = use_geo_token
        self.use_det_flag = use_det_flag
        self.output_percomp = output_percomp
        self.q_granularity = q_granularity
        self._out_per_hand = q_granularity if output_percomp else 1
        self.d_model = d_model

        in_dim = hand_dim + visual_dim + cam_feat_dim + da3_depth_dim + int(use_det_flag)
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.geo_token = (
            nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
            if use_geo_token else None
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.output_mode = output_mode
        head_layers = [
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, self._out_per_hand),
        ]
        if output_mode == "q":
            head_layers.append(nn.Sigmoid())
        self.q_head = nn.Sequential(*head_layers)

    def _forward_single_hand(self, hand_feats: Tensor, conf: Tensor) -> Tensor:
        x = torch.cat([hand_feats, conf], dim=-1) if self.use_det_flag else hand_feats
        x = self.input_proj(x)
        if self.use_geo_token:
            geo = self.geo_token.expand(x.shape[0], 1, self.d_model)
            tokens = self.encoder(torch.cat([geo, x], dim=1))
            x = tokens[:, 1:, :] + tokens[:, 0:1, :]
        else:
            x = self.encoder(x)
        q = self.q_head(x)
        if self.output_mode == "error":
            return q
        return q * (conf > 0.1).float()

    def forward(
        self,
        mano_proposal: Tensor,
        raw_confidence: Tensor,
        visual_feats: Tensor = None,
        cam_feats: Tensor = None,
        da3_depth_signal: Tensor = None,
    ) -> Tensor:
        """Return the left hand's quality groups followed by the right hand's."""
        B, T, D = mano_proposal.shape
        if D != 2 * self.hand_dim:
            raise ValueError(f"Expected {2 * self.hand_dim} proposal features, got {D}")
        parts_L = [mano_proposal[:, :, :self.hand_dim]]
        parts_R = [mano_proposal[:, :, self.hand_dim:]]
        if self.visual_dim > 0:
            if visual_feats is None:
                visual_feats = torch.zeros(B, T, self.visual_dim, device=mano_proposal.device)
            parts_L.append(visual_feats)
            parts_R.append(visual_feats)
        if self.cam_feat_dim > 0:
            if cam_feats is None:
                cam_feats = torch.zeros(B, T, self.cam_feat_dim, device=mano_proposal.device)
            parts_L.append(cam_feats)
            parts_R.append(cam_feats)
        if self.da3_depth_dim > 0:
            if da3_depth_signal is None:
                raise ValueError("The released Quality Network requires da3_depth_signal")
            parts_L.append(da3_depth_signal[:, :, 0, :])
            parts_R.append(da3_depth_signal[:, :, 1, :])
        left_input = torch.cat(parts_L, dim=-1)
        right_input = torch.cat(parts_R, dim=-1)
        q_L = self._forward_single_hand(left_input, raw_confidence[:, :, 0:1])
        q_R = self._forward_single_hand(right_input, raw_confidence[:, :, 1:2])
        return torch.cat([q_L, q_R], dim=-1)


class QualityGatedConditioning(nn.Module):
    """Fuse each hand's motion observation with its projected visual features."""

    def __init__(self, hand_dim: int = SINGLE_HAND_DIM, d_model: int = 512,
                 head_feat_dim: int = 9, scene_feat_dim: int = 0,
                 action_feat_dim: int = 0, vit_dim: int = 0,
                 vit_obs_mode: str = "",
                 visual_proj_dim: int = 0,
                 vit_missing_token: bool = False,
                 disable_cond_gate: bool = False,
                 disable_obs_attn_bias: bool = False,
                 use_skeleton_token: bool = False,
                 skeleton_num_freqs: int = 6):
        super().__init__()
        if action_feat_dim or use_skeleton_token:
            raise ValueError("Released conditioning does not use action or skeleton tokens")
        if vit_obs_mode not in ("", "perhand"):
            raise ValueError("Released visual conditioning uses vit_obs_mode='perhand'")
        if vit_missing_token and vit_obs_mode != "perhand":
            raise ValueError("vit_missing_token requires vit_obs_mode='perhand'")
        self.hand_dim = hand_dim
        self.vit_dim = vit_dim
        self.vit_obs_mode = vit_obs_mode
        self.disable_cond_gate = disable_cond_gate

        if vit_dim > 0 and vit_obs_mode == "perhand":
            self.vit_ln = nn.LayerNorm(vit_dim)
            self.visual_proj = nn.Linear(vit_dim, visual_proj_dim) if visual_proj_dim > 0 else None
            vit_in = visual_proj_dim if visual_proj_dim > 0 else vit_dim
            self.obs_encoder = nn.Sequential(
                nn.Linear(hand_dim + vit_in, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            self.vit_missing = (
                nn.Parameter(torch.randn(vit_in) * 0.02)
                if vit_missing_token else None
            )
        else:
            self.obs_encoder = nn.Sequential(
                nn.Linear(hand_dim, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
        self.null_token = nn.Parameter(torch.randn(d_model) * 0.02)

    def forward(self, mano_proposal: Tensor, q_t: Tensor,
                vit_feats_perhand: Tensor = None, vit_feats: Tensor = None):
        """Return observation tokens for the left and right hands, and no attention bias."""
        B, T, _ = mano_proposal.shape
        left_prop = mano_proposal[:, :, :self.hand_dim]
        right_prop = mano_proposal[:, :, self.hand_dim:]
        if self.vit_obs_mode == "perhand" and self.vit_dim > 0:
            if vit_feats_perhand is None:
                raise ValueError("vit_obs_mode='perhand' requires vit_feats_perhand (B,T,2,vit_dim)")
            raw_L = vit_feats_perhand[:, :, 0, :]
            raw_R = vit_feats_perhand[:, :, 1, :]
            v_L = self.vit_ln(raw_L)
            v_R = self.vit_ln(raw_R)
            if self.visual_proj is not None:
                v_L = self.visual_proj(v_L)
                v_R = self.visual_proj(v_R)
            if self.vit_missing is not None:
                miss = self.vit_missing.view(1, 1, -1)
                pres_L = raw_L.abs().amax(dim=-1, keepdim=True) > 1e-6
                pres_R = raw_R.abs().amax(dim=-1, keepdim=True) > 1e-6
                v_L = torch.where(pres_L, v_L, miss.expand_as(v_L))
                v_R = torch.where(pres_R, v_R, miss.expand_as(v_R))
            left_embed = self.obs_encoder(torch.cat([left_prop, v_L], dim=-1))
            right_embed = self.obs_encoder(torch.cat([right_prop, v_R], dim=-1))
        else:
            left_embed = self.obs_encoder(left_prop)
            right_embed = self.obs_encoder(right_prop)

        if self.disable_cond_gate:
            return torch.cat([left_embed, right_embed], dim=1), None
        if q_t.shape[-1] != 2:
            raise ValueError("Quality gating requires one quality value per hand")
        q_left = q_t[:, :, :1]
        q_right = q_t[:, :, 1:]
        null_embed = self.null_token.view(1, 1, -1).expand(B, T, -1)
        left_gated = q_left * left_embed + (1 - q_left) * null_embed
        right_gated = q_right * right_embed + (1 - q_right) * null_embed
        return torch.cat([left_gated, right_gated], dim=1), None
