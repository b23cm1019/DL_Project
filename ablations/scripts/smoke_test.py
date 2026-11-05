"""End-to-end GPU smoke test of every ablation before the real runs are launched.

Every training config runs a few optimizer updates at its REAL batch size (so out-of-memory shows
up here), including validation, checkpoint saving and auto-resume. Chains (Stage 1 -> Stage 2 ->
cache -> Stage 3) use the actual outputs of the previous smoke step, and every evaluation /
probe / table script runs on the results. A real trained checkpoint (Plan A Stage 3b) checks
that the test-time probes change the output.

Everything is written under <root> (default $OUT_ROOT/ablations_smoke); the real ablation folders
are not touched. Large checkpoints are deleted at the end unless --keep.

  python ablations/scripts/smoke_test.py            (run inside ablations/slurm/smoke.sh)
"""
import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import threading
import time
import traceback
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from basicsr.utils.options import ordered_yaml  # noqa: E402

LOADER, DUMPER = ordered_yaml()
CFG = ROOT / "ablations" / "configs"
PY = sys.executable
ASSET_PY = os.environ.get("ASSET_PY", "/scratch/p26cs0003/krish/envs/lsdir_midas_assets_py310/bin/python")
TORCH_HOME = os.environ.get("TORCH_HOME_MIDAS", "/scratch/p26cs0003/krish/severity_cdd11_lsdir_outdoor_v1/torch_cache")
PLAN_A = Path("/scratch/p26cs0003/krish/experiments/SeverityFull_Stage3b/models")
DCPT_200K = "/scratch/p26cs0003/krish/experiments/DCPT_NAFNet_CDD11_Finetune_750k/models/net_g_200000.pth"

STAGE3_PLAIN = ["s3_ref_planb_200k", "s3_c2_constant", "s3_s1_typeonly", "s3_k1_nocal", "s3_k2_nogate",
                "s3_k3_nocal_nogate", "s3_k4_nokl", "s3_k5_fixednull", "s3_f1_noskip", "s3_f2_middleonly",
                "s3_f3_shift", "s3_f3_prompt", "s3_f4_randinit", "s3_s4_shared"]
CHAINS = {  # key: (has its own Stage 1, Stage 3 configs)
    "s5_categorical": (True, ["s3_s5_categorical"]),
    "s5_scalar": (True, ["s3_s5_scalar"]),
    "s10_middleonly": (True, ["s3_s10_middleonly"]),
    "t5_inprotocol": (True, ["s3_t5_inprotocol"]),
    "x3_holdout_lhs": (False, ["s3_x3_holdout_lhs", "s3_x3_holdout_lhs_typeonly"]),
}
ALL_PROBES = ["base", "shuffle", "null", "sev_up", "sev_down", "noise0.5", "noise1.0", "noise2.0", "oracle_type",
              "oracle_full"]

RESULTS = []


def record(step, ok, seconds, peak=None, note=""):
    RESULTS.append({"step": step, "ok": bool(ok), "seconds": round(seconds, 1),
                    "gpu_peak_gb": None if peak is None else round(peak, 2), "note": note})
    mem = "" if peak is None else f", GPU peak {peak:.1f} GB"
    print(f"[{'PASS' if ok else 'FAIL'}] {step} ({seconds:.0f}s{mem}) {note}", flush=True)
    return ok


class GpuPeak:
    """Peak GPU memory in use during a subprocess (sampled every second with nvidia-smi)."""

    def __init__(self):
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        self.index = visible.split(",")[0] if visible else None

    def _read(self):
        cmd = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
        if self.index is not None:
            cmd += ["-i", self.index]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=15).stdout.split()
            return float(out[0]) / 1024 if out else 0.0
        except Exception:
            return 0.0

    def _loop(self):
        while not self.stop:
            self.peak = max(self.peak, self._read())
            time.sleep(1)

    def __enter__(self):
        self.base = self._read()
        self.peak, self.stop = self.base, False
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop = True
        self.thread.join()

    @property
    def used(self):
        return max(self.peak - self.base, 0.0)


def run(cmd, log, env=None, timeout=5400):
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as handle:
        handle.write("$ " + " ".join(map(str, cmd)) + "\n")
        handle.flush()
        try:
            return subprocess.run(list(map(str, cmd)), stdout=handle, stderr=subprocess.STDOUT, cwd=ROOT,
                                  env=env, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            handle.write("\n[smoke] TIMEOUT\n")
            return -9


def tail(log, n=6):
    try:
        lines = [l for l in Path(log).read_text(errors="ignore").splitlines() if l.strip()]
        return " | ".join(lines[-n:])[-600:]
    except FileNotFoundError:
        return "(no log)"


class Smoke:
    def __init__(self, args):
        self.args = args
        self.root = Path(args.root)
        self.shared = args.shared
        self.logs = self.root / "logs"

    # ------------------------------------------------------------------ configs
    def render_text(self, template):
        return ((CFG / f"{template}.yml").read_text().replace("@OUT_ROOT@", str(self.root))
                .replace("@SHARED_ROOT@", self.shared))

    def load(self, template):
        return yaml.load(self.render_text(template), Loader=LOADER)

    def dump(self, opt, name):
        path = self.root / "configs" / f"{name}.yml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.dump(opt, Dumper=DUMPER, default_flow_style=False, sort_keys=False))
        return path

    def small_manifest(self, path, tag, per_group=2):
        rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
        seen, kept = defaultdict(int), []
        for row in rows:
            group = row.get("setting") or row.get("composition") or "_".join(row.get("active_degradations", []))
            seen[group] += 1
            if seen[group] <= per_group:
                kept.append(row)
        out = self.root / "small_manifests" / f"{tag}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(json.dumps(r) + "\n" for r in kept))
        return str(out)

    # ------------------------------------------------------------------ steps
    def check_shell_render(self):
        start = time.time()
        out = self.root / "rendered_by_shell"
        script = (f"set -euo pipefail; PROJECT_ROOT={ROOT}; OUT_ROOT={out}; SHARED_ROOT={self.shared}; "
                  f"source {ROOT}/ablations/slurm/_render.sh; for f in {CFG}/*.yml; do render_config $(basename $f); done")
        rc = run(["bash", "-c", script], self.logs / "render_shell.log")
        bad = []
        for path in sorted((out / "ablations" / "configs").glob("*.yml")):
            try:
                yaml.load(path.read_text(), Loader=LOADER)
            except Exception as error:  # noqa: BLE001
                bad.append(f"{path.name}: {error}")
        count = len(list((out / "ablations" / "configs").glob("*.yml")))
        return record("launcher config rendering (all templates)", rc == 0 and not bad and count == len(list(CFG.glob("*.yml"))),
                      time.time() - start, note=f"{count} rendered" + (f"; errors: {bad}" if bad else ""))

    def graded(self):
        start = time.time()
        env = dict(os.environ, TORCH_HOME=TORCH_HOME)
        out = self.root / "ablations" / "graded_cdd11"
        rc = run([ASSET_PY, "ablations/scripts/build_graded_testset.py", "--out", out, "--num_images", "2"],
                 self.logs / "graded.log", env=env)
        manifest = out / "manifest.jsonl"
        n = len(manifest.read_text().splitlines()) if manifest.is_file() else 0
        return record("graded test set (2 images x 11 x 4)", rc == 0 and n == 88, time.time() - start,
                      note=f"{n} records" if rc == 0 else tail(self.logs / "graded.log"))

    def t5_data(self):
        """Plan + assets as the real t5_assets.sh, then render a small slice of the plan."""
        start = time.time()
        t5 = self.root / "ablations" / "t5_inprotocol"
        env = dict(os.environ, TORCH_HOME=TORCH_HOME)
        rc = run([ASSET_PY, "ablations/scripts/build_inprotocol_plan.py", "--out_root", t5], self.logs / "t5_plan.log", env=env)
        if rc != 0:
            return record("T5 render plan", False, time.time() - start, note=tail(self.logs / "t5_plan.log"))
        rc = run([ASSET_PY, "scripts/prepare_lsdir_severity_assets.py",
                  "--source-manifest", t5 / "sources" / "train_sources.jsonl",
                  "--source-manifest", t5 / "sources" / "val_sources.jsonl", "--output-root", t5 / "assets",
                  "--rain-mask", "/csehome/p26cs0003/krish/projects/OneRestore/img_file/rain_mask.jpg",
                  "--snow-mask", "/csehome/p26cs0003/krish/projects/OneRestore/img_file/snow_mask.png",
                  "--max-long-side", "512", "--min-short-side", "128", "--device", "cuda"],
                 self.logs / "t5_assets.log", env=env, timeout=7200)
        if rc != 0:
            return record("T5 assets (MiDaS on 1,183 CDD-11 clean scenes)", False, time.time() - start,
                          note=tail(self.logs / "t5_assets.log"))
        small = self.root / "t5_small_plans"
        small.mkdir(parents=True, exist_ok=True)
        for split, count in (("train", 352), ("val", 110)):
            rows = (t5 / "plans" / f"{split}.jsonl").read_text().splitlines()[:count]
            (small / f"{split}.jsonl").write_text("\n".join(rows) + "\n")
        rc = run([PY, "scripts/render_lsdir_cdd11_plan.py", "--plan", small / "train.jsonl", "--plan", small / "val.jsonl",
                  "--asset-manifest", t5 / "assets" / "source_assets.jsonl", "--output-root", t5 / "data",
                  "--shard-index", "0", "--num-shards", "1"], self.logs / "t5_render.log")
        if rc != 0:
            return record("T5 render (slice)", False, time.time() - start, note=tail(self.logs / "t5_render.log"))
        for split in ("train", "val"):  # stand-in for the finalizer (it insists on all 80k / 15k renders)
            part = t5 / "data" / "manifests" / "parts" / f"{split}.part_00.jsonl"
            rows = sorted((json.loads(l) for l in part.read_text().splitlines() if l.strip()), key=lambda r: r["plan_ordinal"])
            (t5 / "data" / "manifests" / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        comps = {json.loads(l)["composition"] for l in (t5 / "data" / "manifests" / "train.jsonl").read_text().splitlines()}
        return record("T5 plan + assets + render (352 train / 110 val renders)", len(comps) == 11, time.time() - start,
                      note=f"{len(comps)} compositions in the rendered slice")

    def train(self, template, iters=None, resume_to=None):
        """Few-update training run of one template at its real batch size."""
        iters = iters or self.args.iters
        start = time.time()
        try:
            opt = self.load(template)
        except Exception:  # noqa: BLE001
            return record(f"train {template}", False, time.time() - start, note=traceback.format_exc(limit=1))
        name = opt["name"]
        is_dcpt = opt.get("model_type") == "SRModel"
        opt["train"]["total_iter"] = iters
        opt["logger"]["print_freq"] = 1
        opt["logger"]["save_checkpoint_freq"] = 2
        opt["val"]["val_freq"] = 10 ** 9 if is_dcpt else 2
        for key, dataset in opt["datasets"].items():
            if key.startswith("val") and "manifest" in dataset:
                if not Path(dataset["manifest"]).is_file():
                    return record(f"train {template}", False, time.time() - start, note=f"missing {dataset['manifest']}")
                dataset["manifest"] = self.small_manifest(dataset["manifest"], f"{name}_{key}")
        config = self.dump(opt, template)
        entry = "basicsr/all_in_one_train.py" if is_dcpt else "ablations/train.py"
        port = 20000 + random.randint(0, 30000)
        log = self.logs / f"train_{template}.log"
        cmd = [PY, "-m", "torch.distributed.run", "--nproc_per_node=1", f"--master-port={port}", entry, "-opt", config,
               "--launcher", "pytorch", "--auto_resume"]
        with GpuPeak() as gpu:
            rc = run(cmd, log)
        models = Path(opt["path"]["experiments_root"]) / "models"
        text = log.read_text(errors="ignore")
        nan = any("nan" in line.lower() for line in text.splitlines() if "l_" in line and "iter:" in line)
        ok = rc == 0 and (models / "net_g_latest.pth").is_file() and "End of training" in text and not nan
        note = "OOM" if "out of memory" in text.lower() else ("NaN loss" if nan else "")
        if not ok and not note:
            note = tail(log)
        if ok and not is_dcpt and "validation" not in text.lower():
            note = "no validation line found"
        record(f"train {template} ({iters} updates)", ok, time.time() - start, gpu.used, note)
        if ok and resume_to:
            start = time.time()
            opt["train"]["total_iter"] = resume_to
            config = self.dump(opt, template)
            log2 = self.logs / f"train_{template}_resume.log"
            resume_cmd = [PY, "-m", "torch.distributed.run", "--nproc_per_node=1",
                          f"--master-port={20000 + random.randint(0, 30000)}", entry, "-opt", config,
                          "--launcher", "pytorch", "--auto_resume"]
            with GpuPeak() as gpu:
                rc = run(resume_cmd, log2)
            text2 = log2.read_text(errors="ignore")
            resumed = f"iter: {iters}" in text2 and "Resuming training" in text2
            record(f"resume {template} ({iters} -> {resume_to})", rc == 0 and resumed, time.time() - start, gpu.used,
                   "" if resumed else tail(log2))
        return ok

    def cache(self, template):
        start = time.time()
        opt = self.load(template)
        config = self.dump(opt, f"{template}__cache")
        log = self.logs / f"cache_{template}.log"
        rc = run([PY, "ablations/scripts/cache_conditions.py", "--config", config, "--limit", str(self.args.cache_rows)], log)
        manifest = Path(opt["datasets"]["train"]["manifest"])
        ok = rc == 0 and manifest.is_file() and Path(opt["datasets"]["val"]["manifest"]).is_file()
        return record(f"cache {template} ({self.args.cache_rows} rows)", ok, time.time() - start, note="" if ok else tail(log))

    def evaluate(self, tag, ckpt, template=None, probes=("base",), testsets=("cdd11", "graded"), limit=1):
        start = time.time()
        cmd = [PY, "ablations/scripts/eval_ablation.py", "--ckpt", ckpt, "--tag", tag, "--testsets", *testsets,
               "--probes", *probes, "--out_dir", self.root / "eval", "--limit_per_setting", str(limit),
               "--graded_manifest", self.root / "ablations" / "graded_cdd11" / "manifest.jsonl"]
        if template:
            opt = self.load(template)
            cmd += ["--config", self.dump(opt, f"{template}__eval")]
        else:
            cmd += ["--mode", "dcpt"]
        log = self.logs / f"eval_{tag}.log"
        with GpuPeak() as gpu:
            rc = run(cmd, log)
        files = [self.root / "eval" / tag / f"{t}__{p}.json" for t in testsets for p in probes
                 if not (p == "oracle_full" and t != "graded")]
        missing = [f.name for f in files if not f.is_file()]
        return record(f"eval {tag} ({len(probes)} probe(s))", rc == 0 and not missing, time.time() - start, gpu.used,
                      "" if rc == 0 and not missing else (f"missing {missing} " + tail(log)))

    def probes_change_output(self, tag):
        """On a TRAINED model, shuffled / null conditions must change PSNR."""
        root = self.root / "eval" / tag
        values = {}
        for probe in ALL_PROBES:
            path = root / f"cdd11__{probe}.json"
            if path.is_file():
                values[probe] = json.loads(path.read_text())["avg_psnr"]
        base = values.get("base")
        deltas = {k: round(v - base, 3) for k, v in values.items() if k != "base"} if base is not None else {}
        ok = base is not None and abs(deltas.get("shuffle", 0)) > 0.005 and abs(deltas.get("null", 0)) > 0.005
        return record("probes change the output of a trained model (Plan A 3b)", ok, 0,
                      note=f"base {base:.3f} dB; delta vs base: {deltas}" if base is not None else "no results")

    def script(self, name, cmd, check):
        start = time.time()
        log = self.logs / f"{name}.log"
        with GpuPeak() as gpu:
            rc = run(cmd, log)
        ok = rc == 0 and check()
        return record(name, ok, time.time() - start, gpu.used, "" if ok else tail(log))

    # ------------------------------------------------------------------ main
    def rerun(self, templates):
        """Re-test only the given training templates (+ a base evaluation of each Stage 3 result)."""
        graded = self.root / "ablations" / "graded_cdd11" / "manifest.jsonl"
        for template in templates:
            folder = Path(self.load(template)["path"]["experiments_root"])
            if folder.exists():
                shutil.rmtree(folder)
            ok = self.train(template)
            if ok and template.startswith("s3_") and graded.is_file():
                ckpt = folder / "models" / "net_g_latest.pth"
                probes = ("base",) if template == "s3_s4_shared" else ("base", "oracle_type")
                self.evaluate(template.removeprefix("s3_"), ckpt, template, probes)
            if not self.args.keep:
                for sub in ("models", "training_states"):
                    shutil.rmtree(folder / sub, ignore_errors=True)

    def main(self):
        self.root.mkdir(parents=True, exist_ok=True)
        if self.args.only:
            self.rerun(self.args.only)
            return self.summarize("smoke_summary_rerun.json")
        experiments = self.root / "ablations" / "experiments"
        if experiments.exists():
            shutil.rmtree(experiments)  # fresh smoke runs (never the real ablation folders)
        self.check_shell_render()
        graded_ok = self.graded()
        t5_ok = self.t5_data()

        # Stage 3 single runs (+ resume of the reference), DCPT hold-out
        for template in STAGE3_PLAIN:
            self.train(template, resume_to=self.args.iters + 2 if template == "s3_ref_planb_200k" else None)
        self.train("dcpt_x3_holdout_lhs")

        # Chains: Stage 1 -> Stage 2 -> cache -> Stage 3
        for key, (own_stage1, stage3s) in CHAINS.items():
            if key == "t5_inprotocol" and not t5_ok:
                record(f"chain {key}", False, 0, note="skipped: T5 data step failed")
                continue
            if own_stage1 and not self.train(f"{key}_stage1",
                                             resume_to=self.args.iters + 2 if key == "s5_categorical" else None):
                continue
            if not self.train(f"{key}_stage2"):
                continue
            if not self.cache(stage3s[0]):
                continue
            for template in stage3s:
                self.train(template)

        # Evaluation scripts on the smoke checkpoints
        if graded_ok:
            for template in STAGE3_PLAIN + [s for _, (_, s3) in CHAINS.items() for s in s3]:
                opt = self.load(template)
                ckpt = Path(opt["path"]["experiments_root"]) / "models" / "net_g_latest.pth"
                if not ckpt.is_file():
                    continue
                probes = ALL_PROBES if template == "s3_ref_planb_200k" else (
                    ("base",) if template == "s3_s4_shared" else ("base", "oracle_type"))
                self.evaluate(template.removeprefix("s3_"), ckpt, template, probes)
            dcpt_smoke = self.root / "ablations" / "experiments" / "DCPT_X3_Holdout_LHS" / "models" / "net_g_latest.pth"
            if dcpt_smoke.is_file():
                self.evaluate("dcpt_x3_holdout", dcpt_smoke)
            self.evaluate("dcpt_200k_real", DCPT_200K)

            # Real trained checkpoint: probes must matter; analyzer + drift; controllability
            plan_a = PLAN_A / "net_g_375000.pth"
            self.evaluate("plan_a_real", plan_a, "s3_ref_planb_200k", ALL_PROBES, limit=2)
            self.probes_change_output("plan_a_real")
            ref_config = self.dump(self.load("s3_ref_planb_200k"), "s3_ref_planb_200k__probe")
            graded = self.root / "ablations" / "graded_cdd11" / "manifest.jsonl"
            analyzer_out = self.root / "probes" / "analyzer.json"
            self.script("analyzer probe S9 + drift S2 (Plan A 50k, 375k)",
                        [PY, "ablations/scripts/analyzer_probe.py", "--config", ref_config, "--graded_manifest", graded,
                         "--synthetic_limit", "30", "--cdd11_limit_per_setting", "2", "--restorer_ckpts",
                         PLAN_A / "net_g_50000.pth", plan_a, "--out", analyzer_out],
                        lambda: len(json.loads(analyzer_out.read_text())["drift"]) == 2)
            control_out = self.root / "probes" / "controllability"
            self.script("controllability X2 (Plan A 375k)",
                        [PY, "ablations/scripts/controllability.py", "--config", ref_config, "--ckpt", plan_a,
                         "--graded_manifest", graded, "--per_cell", "1", "--out", control_out],
                        lambda: (control_out / "controllability.json").is_file())
            table = self.root / "tables" / "smoke_table.md"
            self.script("compare.py table", [PY, "ablations/scripts/compare.py", "--eval_root", self.root / "eval",
                                             "--ref", "ref_planb_200k", "--probe_tags", "plan_a_real", "--out", table],
                        lambda: table.is_file())
        else:
            record("evaluation scripts", False, 0, note="skipped: graded set failed")

        if not self.args.keep:
            for folder in experiments.glob("*"):
                for sub in ("models", "training_states"):
                    shutil.rmtree(folder / sub, ignore_errors=True)
        return self.summarize("smoke_summary.json")

    def summarize(self, filename):
        failed = [r for r in RESULTS if not r["ok"]]
        summary = {"root": str(self.root), "passed": len(RESULTS) - len(failed), "failed": len(failed), "steps": RESULTS}
        (self.root / filename).write_text(json.dumps(summary, indent=2))
        print("\n================ SMOKE SUMMARY ================")
        for r in RESULTS:
            mem = "" if r["gpu_peak_gb"] is None else f"{r['gpu_peak_gb']:5.1f} GB"
            print(f"{'PASS' if r['ok'] else 'FAIL'}  {mem:>8s}  {r['step']}  {r['note'][:160] if not r['ok'] else ''}")
        print(f"\n{summary['passed']} passed, {summary['failed']} failed -> {self.root / filename}")
        return 0 if not failed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=os.path.join(os.environ.get("OUT_ROOT", "/tmp"), "ablations_smoke"))
    parser.add_argument("--shared", default=os.environ.get("SHARED_ROOT", "/scratch/p26cs0003/krish"))
    parser.add_argument("--iters", type=int, default=4, help="optimizer updates per smoke training run")
    parser.add_argument("--cache_rows", type=int, default=120)
    parser.add_argument("--keep", action="store_true", help="keep smoke checkpoints")
    parser.add_argument("--only", nargs="*", default=None,
                        help="re-test only these training templates (names without .yml); needs a finished full run")
    args = parser.parse_args()
    sys.exit(Smoke(args).main())


if __name__ == "__main__":
    main()
