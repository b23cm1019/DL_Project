# Running the proposed pipeline from a second account

This runs two experiments from a second `phd` account (2 running jobs, 3 submitted, 2 GPUs).
Each job uses **one GPU**:

| Slot | Experiment | Stages |
|---|---|---|
| A | **NAFNet, Plan B** (final proposed pipeline) | Stage 3a (50k) → Stage 3b (700k) = 750k, from the existing Stage-2 checkpoint |
| B | **Restormer** (backbone-agnostic proof) | probe → Stage 1 (100k) → Stage 2 (50k) → cache → Stage 3a (50k) → Stage 3b (700k) |

Data and checkpoints are **read in place** from `/scratch/p26cs0003/krish` (read-only).
Everything the new account writes goes to its own `OUT_ROOT` (default `/scratch/<newuser>/sevdcpt`).

---

## 1. Access to grant (run on account p26cs0003)

`/scratch/p26cs0003` currently lets only two named users pass through (ACL). Add the new account:

```bash
setfacl -m u:<NEWUSER>:--x /scratch/p26cs0003
```

Everything below `/scratch/p26cs0003/krish` is already world-readable. The new account reads only:

| Path | Used by |
|---|---|
| `/scratch/p26cs0003/krish/datasets/CDD11/` (train + test) | all stages, env validation |
| `/scratch/p26cs0003/krish/severity_cdd11_lsdir_outdoor_v1/data/` (synthetic images + manifests) | Restormer Stage 1, Stage 2, temperature fit |
| `/scratch/p26cs0003/krish/stage0_official_r4_improved/manifests/` (`train.jsonl`, `val.jsonl`) | Restormer Stage 2, Restormer cache |
| `/scratch/p26cs0003/krish/severity_full/manifests/` (cached NAFNet logits) | NAFNet Plan B 3a/3b |
| `/scratch/p26cs0003/krish/experiments/SeverityDCPT_Stage2_CDDAdapt/models/net_g_latest.pth` and `net_extractor_latest.pth` | NAFNet Plan B init; NAFNet evaluation |

Do not move or delete these while the other account is training.

**Code.** `/csehome/p26cs0003` is private, so share a copy (about 70 MB). This copy must include
the new, uncommitted files:

```bash
# on p26cs0003
mkdir -p /scratch/p26cs0003/krish/share && chmod o+rx /scratch/p26cs0003/krish/share
tar czf /scratch/p26cs0003/krish/share/DL_Project_code.tgz -C /csehome/p26cs0003/krish/projects \
    --exclude=DL_Project/Experiment_Results --exclude=DL_Project/tb_logger --exclude=DL_Project/experiments DL_Project
chmod o+r /scratch/p26cs0003/krish/share/DL_Project_code.tgz
```

```bash
# on the new account
mkdir -p ~/projects && cd ~/projects
tar xzf /scratch/p26cs0003/krish/share/DL_Project_code.tgz
cd DL_Project && mkdir -p portable_logs
```

## 2. Build the Python environments (new account, once)

The venvs cannot be shared (they live in the private home). Build both on a GPU node:
`dl_project` (cu118) is used for A6000/A30, and `dl_project_bw` (cu128) for the RTX PRO 6000 Blackwell.

```bash
cd ~/projects/DL_Project
srun --partition=phd --nodelist=cn07 --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=02:00:00 --pty bash
bash slurm/portable/install_env.sh     # creates ~/envs/dl_project and ~/envs/dl_project_bw, then validates
exit
```

Defaults can be overridden with environment variables: `OUT_ROOT`, `VENV_BASE`, `SHARED_ROOT`.
Always run `sbatch` **from the repo root**. Jobs use it as `PROJECT_ROOT` and write logs to `./portable_logs/`.

## 3. Launch order

The scheduler allows at most 2 running jobs and 3 submitted (running + pending). The commands
below never exceed that. Choose nodes with `sinfo -N -o "%N %G %T"`. NAFNet needs about
18 GB (an A30 on cn01–cn05 is fine). Restormer needs the probe result.

### Round 1 (submit now: 3 jobs)

```bash
cd ~/projects/DL_Project
# Slot A (RECOMMENDED): NAFNet Plan B as ONE stage, 750k full-network updates from Stage 2,
# L1 only, no condition dropout -> DCPT setup + severity condition only.
sbatch --job-name=pb_naf_full --nodelist=cn05 --export=ALL,CONFIG=nafnet_planb_full750k.yml slurm/portable/train.sh
# (Alternative two-phase version: nafnet_planb_stage3a.yml then nafnet_planb_stage3b.yml)
# Slot B: 15-min Restormer memory probe on the node you intend to use
sbatch --job-name=sevrest_probe --nodelist=cn07 slurm/portable/probe_restormer.sh
```

Check the probe: `tail portable_logs/sevrest_probe_*.out`. It must end with `RESTORMER_1GPU_PROBE_OK`.
If it prints `PROBE_TOO_BIG`, re-run it on a larger GPU (A6000 48 GB or RTX PRO 6000 96 GB on cn07).

### Round 2 (after the probe finishes)

```bash
R1=$(sbatch --parsable --job-name=rest_s1 --nodelist=<probe-ok node> --export=ALL,CONFIG=restormer_stage1_synthetic_1gpu.yml slurm/portable/train.sh)
```

### Round 3 onwards (one Restormer job pending at a time)

Each time the previous Restormer job is running, queue the next one with `--dependency=afterok:<previous id>`:

```bash
R2=$(sbatch --parsable --dependency=afterok:$R1 --job-name=rest_s2  --nodelist=<node> --export=ALL,CONFIG=restormer_stage2_cdd_adapt_1gpu.yml slurm/portable/train.sh)
RC=$(sbatch --parsable --dependency=afterok:$R2 --job-name=rest_cache --nodelist=<node> slurm/portable/cache.sh)
R3=$(sbatch --parsable --dependency=afterok:$RC --job-name=rest_3a  --nodelist=<node> --export=ALL,CONFIG=restormer_planb_stage3a.yml slurm/portable/train.sh)
sbatch --dependency=afterok:$R3 --job-name=rest_3b --nodelist=<node> --export=ALL,CONFIG=restormer_planb_stage3b.yml slurm/portable/train.sh
```

Every training job validates its inputs before using the GPU (missing manifest or checkpoint means
an immediate, clear error), renders its config into `OUT_ROOT/configs/`, and **auto-resumes**.
After a wall-time limit or a crash, re-submit the same command.

## 4. What each run does

| Config (`options/cdd_experiments/portable/`) | Batch | Iterations | Trainable | Init |
|---|---|---|---|---|
| `nafnet_planb_stage3a.yml` | 32 | 50k | decoder + FiLM + calibrator (trunk 62.6M frozen) | Stage-2 NAFNet |
| `nafnet_planb_stage3b.yml` | 32 | 700k | everything | 3a @ 50k |
| `restormer_stage1_synthetic_1gpu.yml` | 32 | 100k | Restormer + extractor | scratch |
| `restormer_stage2_cdd_adapt_1gpu.yml` | 16 = 8 CDD + 8 synthetic | 50k | encoder + extractor | Restormer S1 |
| `restormer_planb_stage3a.yml` | 32 | 50k | decoder + FiLM + calibrator (trunk 18.2M frozen) | Restormer S2 |
| `restormer_planb_stage3b.yml` | 32 | 700k | everything | 3a @ 50k |

The batch sizes are the NAFNet arm's global batches on one GPU. The two-GPU Restormer files
(`*_2gpu.yml`) are untouched. Stage 3a: AdamW 3e-4, constant LR after a 2k warmup, FiLM LR ×5,
condition dropout ramped to 10%. Stage 3b: AdamW 3e-4 cosine to 1e-6, 5k warmup, condition LR ×1.
Both: grad clip 0.01, EMA 0.999, L1 + 1.0·FFT + 0.01·KL, full-image cached condition, per-category
PSNR/SSIM validation on 220 test images.

**Time.** NAFNet 3b at about 0.6 s/iteration takes about 4.9 days, which is inside the 5 d 18 h limit.
The Restormer speed is unknown until Stage 1 starts. The deadline may only allow Restormer
Stages 1–2 (plus 3a), which already show that the extractor and conditioning work on a second backbone.

## 5. Monitoring and evaluation

```bash
grep "iter:" portable_logs/pb_naf_3b_*.err | tail -3
grep "Stage 3b validation\|Severity-aware validation" portable_logs/*.err
```

Final evaluation on the full 2,200-image CDD-11 test set (one GPU, about 20 minutes):

```bash
python scripts/evaluate_cdd11_full.py --mode full --tag naf_planb_final \
  --ckpt $OUT_ROOT/experiments/SevFull_PlanB_NAFNet_Stage3b/models/net_g_700000.pth
python scripts/evaluate_cdd11_full.py --mode full --backbone restormer --tag rest_planb_final \
  --ckpt $OUT_ROOT/experiments/SevFull_PlanB_Restormer_Stage3b/models/net_g_<iter>.pth \
  --generator $OUT_ROOT/experiments/SeverityRestormer_Stage2_CDDAdapt_1gpu/models/net_g_latest.pth \
  --extractor $OUT_ROOT/experiments/SeverityRestormer_Stage2_CDDAdapt_1gpu/models/net_extractor_latest.pth
```

Add `--out_dir $OUT_ROOT/eval`. The default output folder belongs to p26cs0003 and is not writable here.
