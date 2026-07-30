import torch
from torch import nn

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACT
from lerobot.utils.constants import ACTION, OBS_DINO_FEATURES, OBS_IMAGES, OBS_STATE


class _FeatureMapBackbone(nn.Module):
    """Treat its input as an already-produced visual feature map."""

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"feature_map": x}


def test_cached_dino_feature_maps_bypass_only_the_backbone():
    """Cached projection inputs must be equivalent to live backbone outputs."""
    camera_keys = [f"{OBS_IMAGES}.cam0", f"{OBS_IMAGES}.cam1"]
    config = ACTConfig(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(3,)),
            **{key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 16, 16)) for key in camera_keys},
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
        vision_backbone="resnet18",
        pretrained_backbone_weights=None,
        dim_model=32,
        n_heads=4,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_decoder_layers=1,
        n_vae_encoder_layers=1,
        chunk_size=2,
        n_action_steps=2,
        use_vae=False,
        dropout=0.0,
    )
    model = ACT(config).eval()
    # ResNet18 normally returns 512 channels.  Replacing it with this tiny
    # deterministic stub isolates the cache branch from DINO/timm downloads.
    model.backbone = _FeatureMapBackbone()

    batch_size = 2
    live_feature_maps = [torch.randn(batch_size, 512, 2, 3) for _ in camera_keys]
    state = torch.randn(batch_size, 3)
    live_batch = {OBS_IMAGES: live_feature_maps, OBS_STATE: state}
    cached_batch = {
        OBS_DINO_FEATURES: torch.stack(live_feature_maps, dim=1),
        OBS_STATE: state,
    }

    with torch.no_grad():
        live_actions, _ = model(live_batch)
        cached_actions, _ = model(cached_batch)

    torch.testing.assert_close(cached_actions, live_actions, rtol=0, atol=0)
