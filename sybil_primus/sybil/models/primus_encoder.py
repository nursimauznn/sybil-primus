import torch
import torch.nn as nn

# Prefix of the Primus weights inside each model's safetensors file
PRIMUS_PREFIX = {
    "nv": "model.vision3d.sub_vision.",
    "colipri": "image_encoder.backbone.",
}

# COLIPRI (RAS) -> NV-Reason-CT (SPL): permutation of the 3 spatial axes, then flips.
# Found on 36 NLST cases in task1/sunucu_kurulum/on_isleme_karsilastir.py
RAS_TO_SPL = ((2, 1, 0), (False, True, True))

PRIMUS_DIM = 864


def reorient(x, transform):
    """Permute and flip the last three (spatial) axes."""
    order, flips = transform
    lead = list(range(x.ndim - 3))
    x = x.permute(*lead, *[len(lead) + i for i in order])
    axes = [len(lead) + i for i, f in enumerate(flips) if f]
    return x.flip(axes) if axes else x


def reorient_inverse(x, transform):
    order, flips = transform
    lead = x.ndim - 3
    axes = [lead + i for i, f in enumerate(flips) if f]
    if axes:
        x = x.flip(axes)
    inverse = sorted(range(3), key=lambda i: order[i])
    return x.permute(*range(lead), *[lead + i for i in inverse])


def build_primus():
    """Same settings as NV-Reason-CT model.py and task1/sunucu_kurulum/resmi_kontrol.py."""
    from dynamic_network_architectures.architectures.primus import Primus

    model = Primus(
        input_channels=1, num_classes=1, eva_depth=16, eva_numheads=12,
        embed_dim=PRIMUS_DIM, patch_embed_size=(8, 8, 8), input_shape=(192, 192, 192),
        use_rot_pos_emb=True, use_abs_pos_embed=False, drop_path_rate=0.2,
        init_values=0.1, scale_attn_inner=True, num_register_tokens=0,
    )
    model.up_projection = nn.Identity()
    return model


def load_primus(weights_path, backbone):
    """Reads only the Primus weights from the safetensors file (LLM / text tower are never loaded)."""
    from safetensors import safe_open

    prefix = PRIMUS_PREFIX[backbone]
    model = build_primus()
    with safe_open(str(weights_path), framework="pt") as f:
        state_dict = {k[len(prefix):]: f.get_tensor(k) for k in f.keys() if k.startswith(prefix)}
    if not state_dict:
        raise RuntimeError(f"No weights with prefix '{prefix}' in {weights_path}")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        raise RuntimeError(f"Primus weights missing in {weights_path}: {missing[:5]} ({len(missing)} total)")
    # up_projection is replaced by Identity, so its weights are expected leftovers
    unexpected = [k for k in unexpected if not k.startswith("up_projection.")]
    if unexpected:
        print(f"Warning: unused Primus weights: {unexpected[:5]} ({len(unexpected)} total)")
    return model


class FrozenPrimusEncoder(nn.Module):
    """
    Frozen Primus 3D ViT (NV-Reason-CT or COLIPRI).
    Input : (B, 1, 192, 192, 192), 2 mm isotropic.
            native_input=False: SPL orientation (NV-Reason-CT layout) for both models;
                                for COLIPRI it is turned to RAS before Primus.
            native_input=True : each model's own preprocessing (NV: SPL, COLIPRI: RAS), as in the design.
    Output: (B, 864, 24, 24, 24) always in SPL orientation, so dim 2 is the axial slice axis Sybil expects.
    """

    def __init__(self, weights_path, backbone, native_input=False):
        super(FrozenPrimusEncoder, self).__init__()
        self.backbone = backbone
        self.native_input = native_input
        self.primus = load_primus(weights_path, backbone)
        self.primus.requires_grad_(False)
        self.primus.eval()

    def train(self, mode=True):
        # drop_path_rate=0.2 must never be active: keep Primus in eval mode even during training
        super(FrozenPrimusEncoder, self).train(mode)
        self.primus.eval()
        return self

    @torch.no_grad()
    def forward(self, x):
        if self.backbone == "colipri" and not self.native_input:
            x = reorient_inverse(x, RAS_TO_SPL).contiguous()
        with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16):
            features = self.primus(x)
        features = features.float()
        if self.backbone == "colipri":
            features = reorient(features, RAS_TO_SPL).contiguous()
        return features


class FeatureAdapter(nn.Sequential):
    """864 -> 512 bridge so the original Sybil pooling layers can be used unchanged."""

    def __init__(self, in_dim=PRIMUS_DIM, out_dim=512, groups=32):
        super(FeatureAdapter, self).__init__(
            nn.Conv3d(in_dim, out_dim, kernel_size=1),
            nn.GroupNorm(groups, out_dim),
            nn.GELU(),
        )
