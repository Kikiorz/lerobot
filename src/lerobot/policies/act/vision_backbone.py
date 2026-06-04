#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import torchvision
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from .configuration_act import ACTConfig


class TimmDinoV2FeatureMap(nn.Module):
    """Expose timm DINOv2 patch tokens through ACT's feature_map interface."""

    def __init__(
        self,
        model_name: str,
        *,
        pretrained: bool,
        train_backbone: bool,
        image_size: tuple[int, int] | None,
    ) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                "timm is required for ACT with `vision_backbone=dinov2`. "
                "Install it with `python3 -m pip install 'timm>=1.0.0,<1.1.0'`."
            ) from exc

        create_kwargs = {
            "pretrained": pretrained,
            "num_classes": 0,
            "global_pool": "",
        }
        if image_size is not None:
            create_kwargs["img_size"] = image_size

        try:
            self.model = timm.create_model(model_name, **create_kwargs)
        except TypeError:
            create_kwargs.pop("img_size", None)
            self.model = timm.create_model(model_name, **create_kwargs)

        self.out_channels = int(getattr(self.model, "num_features"))
        self.patch_size = self._resolve_patch_size()
        self.train_backbone = train_backbone

        if not self.train_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.train_backbone:
            self.model.eval()
        return self

    def _resolve_patch_size(self) -> tuple[int, int]:
        patch_embed = getattr(self.model, "patch_embed", None)
        patch_size = getattr(patch_embed, "patch_size", 14)
        if isinstance(patch_size, tuple):
            return int(patch_size[0]), int(patch_size[1])
        return int(patch_size), int(patch_size)

    def _extract_patch_tokens(self, features: Tensor | dict | tuple | list) -> Tensor:
        if isinstance(features, dict):
            for key in ("x_norm_patchtokens", "x_patchtokens", "patch_tokens", "tokens", "x_norm", "x"):
                if key in features:
                    return features[key]
            raise KeyError(f"Could not find DINOv2 patch tokens in keys: {sorted(features)}")

        if isinstance(features, (tuple, list)):
            return features[-1]

        return features

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        tokens = self._extract_patch_tokens(self.model.forward_features(x))

        if tokens.ndim == 4:
            return {"feature_map": tokens}

        if tokens.ndim != 3:
            raise ValueError(f"Expected DINOv2 tokens with 3 or 4 dims, got shape {tuple(tokens.shape)}.")

        patch_h, patch_w = self.patch_size
        grid_h = x.shape[-2] // patch_h
        grid_w = x.shape[-1] // patch_w
        num_patches = grid_h * grid_w

        if tokens.shape[1] >= num_patches:
            tokens = tokens[:, -num_patches:, :]
        else:
            side = int(math.sqrt(tokens.shape[1]))
            if side * side != tokens.shape[1]:
                raise ValueError(
                    f"Cannot infer DINOv2 patch grid from token shape {tuple(tokens.shape)} "
                    f"and input shape {tuple(x.shape)}."
                )
            grid_h = grid_w = side

        feature_map = tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], grid_h, grid_w)
        return {"feature_map": feature_map}


def make_act_vision_backbone(config: ACTConfig) -> tuple[nn.Module, int]:
    if config.vision_backbone == "dinov2":
        backbone = TimmDinoV2FeatureMap(
            config.dinov2_model,
            pretrained=config.dinov2_pretrained,
            train_backbone=config.dinov2_train_backbone,
            image_size=_image_size_from_config(config),
        )
        return backbone, backbone.out_channels

    backbone_model = getattr(torchvision.models, config.vision_backbone)(
        replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
        weights=config.pretrained_backbone_weights,
        norm_layer=FrozenBatchNorm2d,
    )
    backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})
    return backbone, backbone_model.fc.in_features


def _image_size_from_config(config: ACTConfig) -> tuple[int, int] | None:
    if not config.image_features:
        return None

    shape = next(iter(config.image_features.values())).shape
    if len(shape) != 3:
        return None

    return int(shape[-2]), int(shape[-1])
