"""
Checks the Sybil + frozen Primus (NV-Reason-CT / COLIPRI) model before any training.

Usage (on the GPU server, from the primus-analiz folder):
    uv run python sybil_primus/scripts/primus_test.py --nv <nv_reason_ct/model.safetensors> --colipri <colipri/model.safetensors>
Either path can be left out to test only one backbone.
"""
import argparse
import sys
from argparse import Namespace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # use sybil_primus/sybil without installing it

from sybil.models.primus_encoder import RAS_TO_SPL, reorient, reorient_inverse  # noqa: E402
from sybil.models.sybil import SybilNet  # noqa: E402


def check(condition, message):
    print(("  OK    " if condition else "  FAIL  ") + message)
    if not condition:
        check.failed = True


check.failed = False


def test_reorient():
    print("\n[1] Axis transform (no model needed)")
    x = torch.randn(2, 3, 4, 5, 6)
    y = reorient(x, RAS_TO_SPL)
    check(tuple(y.shape) == (2, 3, 6, 5, 4), f"RAS -> SPL shape {tuple(y.shape)}")
    check(torch.equal(reorient_inverse(y, RAS_TO_SPL), x), "inverse gives back the input")


def count(params):
    return sum(p.numel() for p in params)


def test_backbone(backbone, weights, device):
    print(f"\n[{backbone}] {weights}")
    args = Namespace(backbone=backbone, primus_weights=weights, dropout=0.25, max_followup=6)
    model = SybilNet(args).to(device)

    frozen = count(model.image_encoder.parameters())
    trainable = count(p for p in model.parameters() if p.requires_grad)
    adapter = count(model.feature_adapter.parameters())
    print(f"  frozen Primus: {frozen / 1e6:.1f}M   trainable: {trainable / 1e6:.2f}M (adapter {adapter:,})")
    check(abs(frozen / 1e6 - 144.1) < 0.5, "Primus has ~144.1M parameters")
    check(not any(p.requires_grad for p in model.image_encoder.parameters()), "Primus is frozen")
    check(adapter == 864 * 512 + 512 + 2 * 512, "adapter is Conv3d(864->512) + GroupNorm")

    model.train()
    check(not model.image_encoder.primus.training, "Primus stays in eval mode after model.train()")
    check(model.pool.training and model.dropout.training, "Sybil layers are in train mode")

    torch.manual_seed(0)
    x = torch.randn(1, 1, 192, 192, 192, device=device).clamp(-1, 1)
    out = model(x)
    shapes = {k: tuple(out[k].shape) for k in ("activ", "image_attention_1", "volume_attention_1", "logit", "prob")}
    for key, shape in shapes.items():
        print(f"  {key:20s} {shape}")
    check(shapes["activ"] == (1, 512, 24, 24, 24), "activ is (1, 512, 24, 24, 24)")
    check(shapes["image_attention_1"] == (1, 24, 576), "image attention is (1, 24, 24*24)")
    check(shapes["logit"] == (1, 6), "6 yearly risks")
    check(bool(torch.all(out["prob"][0, 1:] >= out["prob"][0, :-1])), "risk does not decrease over years")

    out["logit"].sum().backward()
    check(all(p.grad is None for p in model.image_encoder.parameters()), "no gradient reaches Primus")
    check(model.feature_adapter[0].weight.grad is not None, "adapter receives gradient")
    check(model.prob_of_failure_layer.hazard_fc.weight.grad is not None, "risk layer receives gradient")

    model.eval()
    with torch.no_grad():
        features = model.image_encoder(x)
        a = model(x)["logit"]
        b = model(features)["logit"]
    check(torch.allclose(a, b, atol=1e-5), "precomputed features give the same result (cache mode)")
    if device == "cuda":
        print(f"  peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nv", help="NV-Reason-CT model.safetensors")
    parser.add_argument("--colipri", help="COLIPRI model.safetensors")
    opts = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    test_reorient()
    for backbone, weights in (("nv", opts.nv), ("colipri", opts.colipri)):
        if weights:
            test_backbone(backbone, weights, device)
            torch.cuda.empty_cache()

    print("\nRESULT:", "some checks FAILED" if check.failed else "all checks passed")
    sys.exit(1 if check.failed else 0)


if __name__ == "__main__":
    main()
