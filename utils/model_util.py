"""Construct the networks and sampler used by the released checkpoints."""

from model.hand_dit import HandDiTModel
from model.quality_network import QualityNetworkTransformer, QualityGatedConditioning
from diffusion.flow_matching import FlowMatchingScheduler

HAND_FEAT_DIM = 108
SINGLE_HAND_DIM = 54
HEAD_FEAT_DIM = 9
BETAS_DIM = 20


def create_model_and_scheduler(args):
    if getattr(args, "qn_arch", None) != "transformer_cam_state":
        raise ValueError("Released checkpoints require qn_arch='transformer_cam_state'")
    if int(getattr(args, "q_granularity", 2)) != 2:
        raise ValueError("Released checkpoints use wrist/finger quality for each hand")
    unsupported = (
        "action_feat_dim", "action_in_dit", "use_skeleton_token", "use_mask_signal",
        "presence_head", "q_adaln_perhand", "detection_aware_attn", "dit_depth_cond_dim",
        "qn_cam_ca_dim", "qn_cross_hand_attention", "qn_delta_dim", "qn_depth_drop_abs",
        "qn_geo_layernorm", "qn_kpt2d_dim", "qn_proposal_trans_layernorm", "qn_reproj_dim",
        "qn_scene_feat_dim", "qn_visual_layernorm", "qn_visual_perhand", "qn_visual_proj_dim",
    )
    enabled = [name for name in unsupported if getattr(args, name, False)]
    if enabled:
        raise ValueError(f"Unsupported checkpoint options: {', '.join(enabled)}")

    inner_dim = args.heads * 64
    visual_dim = getattr(args, "visual_dim", 0)
    scene_feat_dim = getattr(args, "scene_feat_dim", 0)
    uses_percomp = (getattr(args, "modality_tokens", False)
                    and getattr(args, "use_percomp_q", False))
    q_adaln_dim = (4 if uses_percomp else 2) if getattr(args, "q_adaln", False) else 0
    # Preserve the serialized attention order, including its unused action slot.
    cross_attn_order = tuple(
        part.strip() for part in getattr(args, "cross_attn_order", "scene,action,head,obs").split(",")
        if part.strip())

    model = HandDiTModel(
        hand_feat_dim=HAND_FEAT_DIM,
        head_feat_dim=HEAD_FEAT_DIM,
        betas_dim=BETAS_DIM,
        num_layers=args.layers,
        num_attention_heads=args.heads,
        attention_head_dim=64,
        zero_init=getattr(args, "zero_init", False),
        disable_crop_embed=getattr(args, "no_crop_embed", False),
        disable_tarope=getattr(args, "no_tarope", False),
        disable_eff_t=getattr(args, "no_eff_t", False),
        scene_feat_dim=scene_feat_dim,
        modality_tokens=getattr(args, "modality_tokens", False),
        q_adaln_dim=q_adaln_dim,
        cross_attn_order=cross_attn_order,
        betas_input_ln=getattr(args, "betas_input_ln", False),
    )

    qn_d_model = getattr(args, "qn_d_model", 256)
    quality_net = QualityNetworkTransformer(
        hand_dim=SINGLE_HAND_DIM,
        visual_dim=visual_dim,
        cam_feat_dim=getattr(args, "qn_cam_feat_dim", 9),
        da3_depth_dim=int(getattr(args, "qn_da3_depth_dim", 0) or 0),
        da3_depth_layernorm=bool(getattr(args, "qn_da3_depth_layernorm", False)),
        use_geo_token=True,
        output_percomp=getattr(args, "qn_output_percomp", False),
        output_mode=getattr(args, "qn_output_mode", "q"),
        use_det_flag=not bool(getattr(args, "qn_no_det_flag", False)),
        d_model=qn_d_model,
        n_heads=getattr(args, "qn_heads", 4),
        n_layers=getattr(args, "qn_layers", 4),
        dim_feedforward=4 * qn_d_model,
        dropout=0.0,
    )

    quality_gate = QualityGatedConditioning(
        hand_dim=SINGLE_HAND_DIM,
        d_model=inner_dim,
        head_feat_dim=HEAD_FEAT_DIM,
        scene_feat_dim=scene_feat_dim,
        vit_dim=visual_dim,
        vit_obs_mode=getattr(args, "vit_obs_mode", ""),
        visual_proj_dim=getattr(args, "dit_visual_proj_dim", 0),
        vit_missing_token=getattr(args, "vit_obs_missing_token", False),
        disable_cond_gate=getattr(args, "no_quality_cond_gate", False),
        disable_obs_attn_bias=getattr(args, "no_obs_attn_bias", False),
    )
    scheduler = FlowMatchingScheduler(sigma_min=1e-4, pred_type=getattr(args, "fm_pred_type", "v"))
    return model, quality_net, quality_gate, scheduler
