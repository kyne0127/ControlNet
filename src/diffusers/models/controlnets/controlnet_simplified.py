# # my_controlnet_simplified.py

# from typing import Any, Dict, List, Optional, Tuple, Union
# import torch
# from torch import nn
# from torch.nn import functional as F

# from diffusers.models.controlnets.controlnet import ControlNetModel
# from ...configuration_utils import ConfigMixin, register_to_config
# from ...utils import BaseOutput, logging
# from ..attention_processor import (
#     ADDED_KV_ATTENTION_PROCESSORS,
#     CROSS_ATTENTION_PROCESSORS,
#     AttentionProcessor,
#     AttnAddedKVProcessor,
#     AttnProcessor,
#     UNet2DConditionModel,
# )
# from ..embeddings import (
#     TextImageProjection,
#     TextImageTimeEmbedding,
#     TextTimeEmbedding,
#     TimestepEmbedding,
#     Timesteps,
# )
# from ..unets.unet_2d_blocks import (
#     UNetMidBlock2D,
#     UNetMidBlock2DCrossAttn,
#     get_down_block,
# )

# logger = logging.get_logger(__name__)


# @dataclass
# class ControlNetOutput(BaseOutput):
#     down_block_res_samples: Tuple[torch.Tensor]
#     mid_block_res_sample: torch.Tensor


# class MyControlNetSimplified(ControlNetModel, ConfigMixin):
#     """
#     ControlNet-XS 스타일로 축소한 버전.
#     """

#     @register_to_config
#     def __init__(
#         self,
#         in_channels: int = 4,
#         conditioning_channels: int = 3,
#         flip_sin_to_cos: bool = True,
#         freq_shift: int = 0,
#         down_block_types: Tuple[str, ...] = (
#             "CrossAttnDownBlock2D",
#             "CrossAttnDownBlock2D",
#             "CrossAttnDownBlock2D",
#             "DownBlock2D",
#         ),
#         mid_block_type: Optional[str] = "UNetMidBlock2DCrossAttn",
#         only_cross_attention: Union[bool, Tuple[bool]] = False,

#         # ↓ XS 수준으로 과감히 축소
#         block_out_channels: Tuple[int, ...] = (64, 128, 256, 256),
#         layers_per_block: int = 1,
#         downsample_padding: int = 1,
#         mid_block_scale_factor: float = 1,
#         act_fn: str = "silu",
#         norm_num_groups: Optional[int] = 8,
#         norm_eps: float = 1e-5,
#         cross_attention_dim: int = 256,
#         transformer_layers_per_block: Union[int, Tuple[int, ...]] = 1,
#         attention_head_dim: Union[int, Tuple[int, ...]] = 4,
#         num_attention_heads: Optional[Union[int, Tuple[int, ...]]] = None,
#         use_linear_projection: bool = False,
#         class_embed_type: Optional[str] = None,
#         addition_embed_type: Optional[str] = None,
#         addition_time_embed_dim: Optional[int] = None,
#         num_class_embeds: Optional[int] = None,
#         upcast_attention: bool = False,
#         resnet_time_scale_shift: str = "default",
#         projection_class_embeddings_input_dim: Optional[int] = None,
#         controlnet_conditioning_channel_order: str = "rgb",

#         # ↓ conditioning 임베딩도 축소
#         conditioning_embedding_out_channels: Optional[Tuple[int, ...]] = (8, 16, 32, 64),

#         global_pool_conditions: bool = False,
#         addition_embed_type_num_heads: int = 64,
#     ):
#         # super 에 축소된 기본값만 넘겨주면, 내부 구조는 그대로 ControlNetModel
#         super().__init__(
#             in_channels=in_channels,
#             conditioning_channels=conditioning_channels,
#             flip_sin_to_cos=flip_sin_to_cos,
#             freq_shift=freq_shift,
#             down_block_types=down_block_types,
#             mid_block_type=mid_block_type,
#             only_cross_attention=only_cross_attention,
#             block_out_channels=block_out_channels,
#             layers_per_block=layers_per_block,
#             downsample_padding=downsample_padding,
#             mid_block_scale_factor=mid_block_scale_factor,
#             act_fn=act_fn,
#             norm_num_groups=norm_num_groups,
#             norm_eps=norm_eps,
#             cross_attention_dim=cross_attention_dim,
#             transformer_layers_per_block=transformer_layers_per_block,
#             attention_head_dim=attention_head_dim,
#             num_attention_heads=num_attention_heads,
#             use_linear_projection=use_linear_projection,
#             class_embed_type=class_embed_type,
#             addition_embed_type=addition_embed_type,
#             addition_time_embed_dim=addition_time_embed_dim,
#             num_class_embeds=num_class_embeds,
#             upcast_attention=upcast_attention,
#             resnet_time_scale_shift=resnet_time_scale_shift,
#             projection_class_embeddings_input_dim=projection_class_embeddings_input_dim,
#             controlnet_conditioning_channel_order=controlnet_conditioning_channel_order,
#             conditioning_embedding_out_channels=conditioning_embedding_out_channels,
#             global_pool_conditions=global_pool_conditions,
#             addition_embed_type_num_heads=addition_embed_type_num_heads,
#         )

#     @classmethod
#     def from_unet(
#         cls,
#         unet: UNet2DConditionModel,
#         controlnet_conditioning_channel_order: str = "rgb",
#         conditioning_embedding_out_channels: Optional[Tuple[int, ...]] = (8, 16, 32, 64),
#         load_weights_from_unet: bool = False,       # ← 기본 동작을 복사하지 않도록 False
#         conditioning_channels: int = 3,
#     ):
#         # super().from_unet 에 load_weights_from_unet=False 만 전달
#         return super().from_unet(
#             unet=unet,
#             controlnet_conditioning_channel_order=controlnet_conditioning_channel_order,
#             conditioning_embedding_out_channels=conditioning_embedding_out_channels,
#             load_weights_from_unet=load_weights_from_unet,
#             conditioning_channels=conditioning_channels,
#         )
