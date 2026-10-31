"""One-GPU memory probe for the Restormer arm at the NAFNet-equivalent batch sizes.

Measures peak memory for:
  stage1  : B=32 clean identity + LQ feature tap + extractor (heaviest of Stage 1/2)
  stage3b : B=32 full fine-tuning of SeverityConditionedFullRestormer (+ EMA copy)
  val     : one full 720x1080 CDD-11 image, inference only
Prints PROBE_OK / PROBE_TOO_BIG per phase. Exit code 3 if any phase does not fit.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from basicsr.archs.feature_severity_extractor_arch import FeatureSeverityExtractor  # noqa: E402
from basicsr.archs.feature_tap_restormer_arch import FeatureTapRestormer  # noqa: E402
from basicsr.archs.severity_conditioned_full_restormer_arch import SeverityConditionedFullRestormer  # noqa: E402
from basicsr.utils.severity_full_analysis import RESTORMER_ARGS, RESTORMER_EXTRACTOR_ARGS  # noqa: E402

HEADROOM_GIB = 1.5


def gib(value):
    return value / 2 ** 30


def run_phase(name, fn, device, total):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    try:
        fn()
        torch.cuda.synchronize(device)
        peak = gib(torch.cuda.max_memory_reserved(device))
        ok = peak + HEADROOM_GIB <= total
        print(f"{'PROBE_OK' if ok else 'PROBE_TOO_BIG'} phase={name} max_reserved_gib={peak:.2f} gpu_total_gib={total:.2f}", flush=True)
        return ok
    except torch.cuda.OutOfMemoryError:
        print(f"PROBE_TOO_BIG phase={name} (CUDA out of memory) gpu_total_gib={total:.2f}", flush=True)
        return False


def main():
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    total = gib(torch.cuda.get_device_properties(device).total_memory)
    print(f"GPU: {torch.cuda.get_device_name(device)} ({total:.1f} GiB)", flush=True)

    def stage1():
        backbone = FeatureTapRestormer(**RESTORMER_ARGS).to(device)
        extractor = FeatureSeverityExtractor(**RESTORMER_EXTRACTOR_ARGS).to(device)
        opt_g = torch.optim.AdamW(backbone.parameters(), lr=3e-4)
        opt_e = torch.optim.AdamW(extractor.parameters(), lr=1e-4)
        for _ in range(2):
            clean = torch.rand(32, 3, 128, 128, device=device)
            lq = torch.rand(32, 3, 128, 128, device=device)
            opt_g.zero_grad(set_to_none=True)
            opt_e.zero_grad(set_to_none=True)
            identity = (backbone(clean) - clean).abs().mean()
            features, latent = backbone(lq, return_degradation_features=True)
            loss = identity + extractor(features, latent).square().mean()
            loss.backward()
            opt_g.step()
            opt_e.step()
        del backbone, extractor, opt_g, opt_e

    def stage3b():
        net = SeverityConditionedFullRestormer(**RESTORMER_ARGS).to(device)
        ema = SeverityConditionedFullRestormer(**RESTORMER_ARGS).to(device).eval()
        opt = torch.optim.AdamW(net.parameters(), lr=3e-4)
        for _ in range(2):
            lq = torch.rand(32, 3, 128, 128, device=device)
            logits = torch.randn(32, 4, 4, device=device)
            opt.zero_grad(set_to_none=True)
            out = net(lq, logits, condition_keep=torch.ones(32, device=device))
            loss = out.abs().mean() + torch.fft.rfft2(out, norm="ortho").abs().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.01)
            opt.step()
        del net, ema, opt

    def val():
        net = SeverityConditionedFullRestormer(**RESTORMER_ARGS).to(device).eval()
        with torch.inference_mode():
            net(torch.rand(1, 3, 720, 1080, device=device), torch.randn(1, 4, 4, device=device))
        del net

    results = [run_phase(n, f, device, total) for n, f in (("stage1", stage1), ("stage3b", stage3b), ("val", val))]
    if all(results):
        print("RESTORMER_1GPU_PROBE_OK", flush=True)
    else:
        print("RESTORMER_1GPU_PROBE_FAILED: use a larger GPU (A6000 48 GB / RTX PRO 6000 96 GB on cn07)", flush=True)
        sys.exit(3)


if __name__ == "__main__":
    main()
