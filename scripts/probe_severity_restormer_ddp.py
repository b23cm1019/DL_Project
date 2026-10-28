"""Two-rank Stage-1-equivalent GPU memory probe for standard Restormer."""

import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from basicsr.archs import build_network


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank)
    backbone = build_network({
        "type": "FeatureTapRestormer", "inp_channels": 3, "out_channels": 3,
        "dim": 48, "num_blocks": [4, 6, 6, 8], "num_refinement_blocks": 4,
        "heads": [1, 2, 4, 8], "ffn_expansion_factor": 2.66, "bias": False,
        "LayerNorm_type": "WithBias", "dual_pixel_task": False, "scale": 1,
        "window_size": 8,
    }).to(device)
    extractor = build_network({
        "type": "FeatureSeverityExtractor", "feature_dims": [48, 96, 192],
        "middle_dim": 384, "hidden_dim": 256, "num_types": 4, "num_bins": 4,
    }).to(device)
    backbone = DDP(backbone, device_ids=[local_rank])
    extractor = DDP(extractor, device_ids=[local_rank])
    optimizer_g = torch.optim.AdamW(backbone.parameters(), lr=3e-4, weight_decay=0)
    optimizer_e = torch.optim.AdamW(extractor.parameters(), lr=1e-4, weight_decay=0)
    torch.cuda.reset_peak_memory_stats(device)

    # B=16 per rank is the proposed B=32 global Stage-1 setting.  This has the
    # same clean identity and LQ encoder/latent extractor graphs as Stage 1.
    for _ in range(2):
        clean = torch.rand(16, 3, 128, 128, device=device)
        lq = torch.rand(16, 3, 128, 128, device=device)
        optimizer_g.zero_grad(set_to_none=True)
        optimizer_e.zero_grad(set_to_none=True)
        identity = (backbone(clean) - clean).abs().mean()
        features, latent = backbone(lq, return_degradation_features=True)
        logits = extractor(features, latent)
        loss = identity + logits.square().mean()
        loss.backward()
        optimizer_g.step()
        optimizer_e.step()
    torch.cuda.synchronize(device)
    allocated = torch.cuda.max_memory_allocated(device)
    reserved = torch.cuda.max_memory_reserved(device)
    values = torch.tensor([allocated, reserved], dtype=torch.float64, device=device)
    dist.all_reduce(values, op=dist.ReduceOp.MAX)
    if dist.get_rank() == 0:
        print(
            "RESTORMER_STAGE1_DDP_B16_PROBE_OK "
            f"max_allocated_gib={values[0].item() / 2**30:.2f} "
            f"max_reserved_gib={values[1].item() / 2**30:.2f}"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
