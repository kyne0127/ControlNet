# my_controlnet_simplified.py
# ControlNet-XS: 채널 축소 + 양방향 feature flow 통합
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders.single_file_model import FromOriginalModelMixin
from ...utils import BaseOutput, logging
from ..attention_processor import (
    ADDED_KV_ATTENTION_PROCESSORS,
    CROSS_ATTENTION_PROCESSORS,
    AttentionProcessor,
    AttnAddedKVProcessor,
    AttnProcessor,
)
from ..embeddings import TextImageProjection, TextImageTimeEmbedding, TextTimeEmbedding, TimestepEmbedding, Timesteps
from ..unets.unet_2d_blocks import (
    UNetMidBlock2D,
    UNetMidBlock2DCrossAttn,
    get_down_block,
)
from diffusers.models.controlnets.controlnet import ControlNetModel
from ..unets.unet_2d_condition import UNet2DConditionModel

logger = logging.get_logger(__name__)

@dataclass
class ControlNetOutput(BaseOutput):
    down_block_res_samples: Tuple[torch.Tensor]
    mid_block_res_sample: torch.Tensor

class ControlNetSimplifiedModel(ControlNetModel, ConfigMixin):
    """
    ControlNet-XS 스타일로 채널 수 축소 및 UNet<->ControlNet 양방향 feature flow 통합
    """

    @register_to_config
    def __init__(
        self,
        in_channels: int = 4,
        conditioning_channels: int = 3,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types: Tuple[str, ...] = (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        mid_block_type: Optional[str] = "UNetMidBlock2DCrossAttn",
        only_cross_attention: Union[bool, Tuple[bool]] = False,
        encoder_hid_dim: Optional[int] = None,
        encoder_hid_dim_type: Optional[str] = None,
        # XS 용 채널 축소
        block_out_channels: Tuple[int, ...] = (64, 128, 256, 256),
        layers_per_block: int = 1,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        act_fn: str = "silu",
        norm_num_groups: Optional[int] = 8,
        norm_eps: float = 1e-5,
        cross_attention_dim: int = 256,
        transformer_layers_per_block: Union[int, Tuple[int, ...]] = 1,
        attention_head_dim: Union[int, Tuple[int, ...]] = 4,
        num_attention_heads: Optional[Union[int, Tuple[int, ...]]] = None,
        use_linear_projection: bool = False,
        class_embed_type: Optional[str] = None,
        addition_embed_type: Optional[str] = None,
        addition_time_embed_dim: Optional[int] = None,
        num_class_embeds: Optional[int] = None,
        upcast_attention: bool = False,
        resnet_time_scale_shift: str = "default",
        projection_class_embeddings_input_dim: Optional[int] = None,
        controlnet_conditioning_channel_order: str = "rgb",
        conditioning_embedding_out_channels: Optional[Tuple[int, ...]] = (8, 16, 32, 64),
        global_pool_conditions: bool = False,
        addition_embed_type_num_heads: int = 64,
    ):
        super().__init__(
            in_channels=in_channels,
            conditioning_channels=conditioning_channels,
            flip_sin_to_cos=flip_sin_to_cos,
            freq_shift=freq_shift,
            down_block_types=down_block_types,
            mid_block_type=mid_block_type,
            only_cross_attention=only_cross_attention,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            downsample_padding=downsample_padding,
            mid_block_scale_factor=mid_block_scale_factor,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            cross_attention_dim=cross_attention_dim,
            transformer_layers_per_block=transformer_layers_per_block,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            encoder_hid_dim=encoder_hid_dim,
            encoder_hid_dim_type=encoder_hid_dim_type,
            use_linear_projection=use_linear_projection,
            class_embed_type=class_embed_type,
            addition_embed_type=addition_embed_type,
            addition_time_embed_dim=addition_time_embed_dim,
            num_class_embeds=num_class_embeds,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
            projection_class_embeddings_input_dim=projection_class_embeddings_input_dim,
            controlnet_conditioning_channel_order=controlnet_conditioning_channel_order,
            conditioning_embedding_out_channels=conditioning_embedding_out_channels,
            global_pool_conditions=global_pool_conditions,
            addition_embed_type_num_heads=addition_embed_type_num_heads,
        )

    @classmethod
    def from_unet(
        cls,
        unet: UNet2DConditionModel,
        controlnet_conditioning_channel_order: str = "rgb",
        conditioning_embedding_out_channels: Optional[Tuple[int, ...]] = (8, 16, 32, 64),
        load_weights_from_unet: bool = False,
        conditioning_channels: int = 3,
    ):
        return super().from_unet(
            unet=unet,
            controlnet_conditioning_channel_order=controlnet_conditioning_channel_order,
            conditioning_embedding_out_channels=conditioning_embedding_out_channels,
            load_weights_from_unet=load_weights_from_unet,
            conditioning_channels=conditioning_channels,
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        controlnet_cond: torch.Tensor,
        unet_down_res: Tuple[torch.Tensor, ...],
        unet_mid_res: torch.Tensor,
        conditioning_scale: float = 1.0,
        return_dict: bool = True,
    ) -> Union[ControlNetOutput, Tuple[Tuple[torch.Tensor, ...], torch.Tensor]]:
        # time embedding
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], device=sample.device)
        t_emb = self.time_proj(timestep).to(sample.dtype)
        emb = self.time_embedding(t_emb)

        # preprocess
        x = self.conv_in(sample)
        cond_emb = self.controlnet_cond_embedding(controlnet_cond)
        x = x + cond_emb
        print(f"Input sample shape: {sample.shape}, x shape after conv_in: {x.shape}, cond_emb shape: {cond_emb.shape}")

        down_ctrl: List[torch.Tensor] = []

        # UNet의 첫 번째(원본 해상도) 피쳐는 skip, 나머지부터 다운블록과 매칭
        for down_block, u_feat in zip(self.down_blocks, unet_down_res[1:]):
            # ControlNet쪽 다운블록 실행
            if hasattr(down_block, "has_cross_attention") and down_block.has_cross_attention:
                x, _ = down_block(x, emb, encoder_hidden_states)
            else:
                x, _ = down_block(x, emb)

            # 이제 x와 u_feat는 같은 해상도·채널이므로 바로 fusion
            print(f"Down block output shape: {x.shape}, unet feature shape: {u_feat.shape}")
            x = x + u_feat
            down_ctrl.append(x)

        # 이후 mid-block fusion
        mid = self.mid_block(x, emb, encoder_hidden_states)
        fused_mid = mid + unet_mid_res

        print(f"Mid block output shape: {mid.shape}, fused_mid shape: {fused_mid.shape}")

        # ─── UNet에 바로 더해줄 fused feature를 곧바로 반환 ───────────────────────
        down_block_res_samples = tuple(down_ctrl)
        mid_block_res_sample = fused_mid

        print(f"Final down_ctrl shapes: {[d.shape for d in down_block_res_samples]}, mid shape: {mid_block_res_sample.shape}")

        if not return_dict:
            return down_block_res_samples, mid_block_res_sample

        return ControlNetOutput(
            down_block_res_samples=down_block_res_samples,
            mid_block_res_sample=mid_block_res_sample,
        )


# 모듈 레벨 덮어쓰기
ControlNetModel = ControlNetSimplifiedModel
__all__ = ["ControlNetModel", "ControlNetSimplifiedModel", "ControlNetOutput"]