"""CPU correctness smoke test for the backbone-agnostic Restormer adapter."""

import torch

from basicsr.archs import build_network


def restormer_options(network_type):
    return {
        "type": network_type,
        "inp_channels": 3,
        "out_channels": 3,
        "dim": 8,
        "num_blocks": [1, 1, 1, 1],
        "num_refinement_blocks": 1,
        "heads": [1, 2, 4, 8],
        "ffn_expansion_factor": 2.0,
        "bias": False,
        "LayerNorm_type": "WithBias",
        "dual_pixel_task": False,
        "scale": 1,
        "window_size": 8,
    }


def main():
    torch.manual_seed(0)
    image = torch.rand(2, 3, 17, 19)  # Deliberately not divisible by eight.
    backbone = build_network(restormer_options("FeatureTapRestormer")).eval()
    conditioned_options = restormer_options("FrozenFeatureConditionedRestormer")
    conditioned_options.update(condition_dim=16, adapter_dim=16)
    conditioned = build_network(conditioned_options).eval()
    conditioned.load_state_dict(backbone.state_dict(), strict=False)
    extractor = build_network({
        "type": "FeatureSeverityExtractor",
        "feature_dims": [8, 16, 32],
        "middle_dim": 64,
        "hidden_dim": 16,
        "num_types": 4,
        "num_bins": 4,
    }).eval()

    with torch.no_grad():
        features, latent = backbone(image, return_degradation_features=True)
        logits = extractor(features, latent)
        condition = logits.softmax(dim=-1).flatten(1)
        unconditioned = backbone(image)
        zero_adapter_output = conditioned(image, condition)

    assert [feature.shape[1] for feature in features] == [8, 16, 32]
    assert latent.shape[1] == 64
    assert logits.shape == (2, 4, 4)
    assert condition.shape == (2, 16)
    assert unconditioned.shape == image.shape
    assert torch.allclose(unconditioned, zero_adapter_output, atol=1e-6, rtol=1e-5)
    print("SEVERITY_RESTORMER_CPU_SMOKE_OK")


if __name__ == "__main__":
    main()
