# Phase-1 component ablations: run plan across three accounts

Written 5 Nov 2026. Goal: fix the final architecture before the CVPR 2027 abstract registration
(10 Nov) by screening each component at **100k updates on Plan B's 750k learning-rate curve**.
Every result is compared with the existing Plan B and DCPT `net_g_100000.pth` checkpoints.

Code: branch `restormer-proposed-portable`, commit `2f9e781` or later (`git log -1 --oneline`).

---

## 1. The ablations

| ID | Config | Change from Plan B | Question it answers | Priority |
|---|---|---|---|---|
| C2 | `s3_c2_constant_100k.yml` | same learned null condition for every image | Is the gain per-image information, or just FiLM capacity? | 1 |
| S1 | `s3_s1_typeonly_100k.yml` | severity flattened, degradation type kept | Does severity help beyond knowing the type? | 1 |
| K1 | `s3_k1_nocal_100k.yml` | calibrator off | Is the calibrator needed? | 1 |
| K2 | `s3_k2_nogate_100k.yml` | entropy gate off | Is the gate needed? (gate stays ~0.998 in Plan B, so expected: no) | 1 |
| K4 | `s3_k4_nokl_100k.yml` | KL weight 0 | Is the KL term needed? | 2 |
| F1 | `s3_f1_noskip_100k.yml` | no FiLM on skip connections | Is the new skip-FiLM needed? | 2 |
| S4 | `s3_s4_shared_100k.yml` | condition from the restorer's own trainable encoder | Is the frozen analysis copy needed? | 2 |
| T3 | `s3_t3_nostage2_100k.yml` | no Stage 2: start from Stage 1, Stage 1 analyzer | Is Stage 2 (CDD-11 adaptation) needed? Needs a ~1 h cache first | 2 |
| F2 | `s3_f2_middleonly_100k.yml` | FiLM at the middle block only | Is conditioning at every scale needed? | 3 |
| F4 | `s3_f4_randinit_100k.yml` | FiLM randomly initialized | Is zero-initialized FiLM needed? | 3 |
| K3 | `s3_k3_nocal_nogate_100k.yml` | calibrator and gate both off | Reserve: only if K1 and K2 are both borderline | — |
| K5 | `s3_k5_fixednull_100k.yml` | fixed uniform null condition | Reserve: only if K2 shows the gate matters | — |

Not in this phase (do not change the architecture): FiLM vs concatenation / prompts, analyzer
design (S5, S10), data source (T5), held-out combinations, seeds, Restormer repeats.

---

## 2. Who runs what

| Account | GPU and limits | Runs, in order | Approx. finish |
|---|---|---|---|
| **p26cs0003** | phd partition; 1 free slot (slot 2 = Restormer Stage 3, do not touch) | **C2 → K2 → T3** | C2 ~6 Nov, K2 ~7 Nov, T3 ~8 Nov |
| **b23cm1019** | btech, A30, 1 running job, 12 h per job | **S1 → K1 → F2 → F4** (each run = 2 jobs) | S1 ~6 Nov, K1 ~7 Nov, F2 / F4 ~8 Nov |
| **akshatmishra** | dedicated A5000 24 GB, no Slurm | **K4 → F1 → S4** | ~7-8 Nov |

Priority-1 runs are spread so all four finish first. Each config runs on exactly one account; never
start the same config on two machines. Times assume ~13 h per run on an A30, ~16-18 h on the A5000.

---

## 3. One-time checks per account

### p26cs0003
Nothing to set up. `OUT_ROOT=/scratch/p26cs0003/krish/sevdcpt`.

### b23cm1019
```bash
cd <b23 repo>
git status                                   # local edits -> git stash
git fetch origin && git checkout restormer-proposed-portable && git pull --ff-only
git log -1 --oneline                         # 2f9e781 or later
ls ablations/configs/*_100k.yml | wc -l      # 12
export OUT_ROOT=<b23 scratch>/sevdcpt        # writable, ~45 GB free for 4 runs
export VENV_BASE=<folder that contains dl_project/>    # only if not ~/envs or ~/krish/envs
ls $VENV_BASE/dl_project/bin/activate
sbatch --test-only -p btech --time 12:00:00 --gres=gpu:1 --cpus-per-task=8 --mem=48G --wrap true
```
Data is read in place from `/scratch/p26cs0003/krish` (access already checked). If `--test-only`
rejects the memory or CPU request, lower `--mem` / `--cpus-per-task` in `ablations/slurm/train.sh`.

### akshatmishra (A5000)
```bash
cd <repo> && git fetch origin && git checkout restormer-proposed-portable && git pull --ff-only
python -c "import torch,cv2,yaml,scipy,tensorboard; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
PYTHONPATH=$PWD python -c "import ablations.register; print('ablations OK')"
S=/DATA/user/akshat_mishra1/krish/sevdcpt_shared
ls $S/datasets/CDD11 $S/experiments/SeverityDCPT_Stage2_CDDAdapt/models   # net_g_latest + net_extractor_latest
wc -l $S/severity_full/manifests/train_full.jsonl                          # 13013
grep -c /scratch/p26cs0003 $S/severity_full/manifests/*.jsonl              # must be 0 (paths rewritten)
```
Missing packages: `pip install -r requirements-cluster.txt`. Free space: ~11 GB per run.

---

## 4. Launch commands

### p26cs0003
```bash
cd /csehome/p26cs0003/krish/projects/DL_Project && export OUT_ROOT=/scratch/p26cs0003/krish/sevdcpt
ablations/submit.sh s3_c2_constant_100k.yml                      # starts when Plan B job 46875 ends
ablations/submit.sh --after <C2 job> s3_k2_nogate_100k.yml       # once 46875 has ended (3-job limit)
# T3, once K2 is queued and C2 has finished:
J=$(sbatch --parsable --dependency=afterany:<K2 job> --job-name=abl_t3_cache --export=ALL,CONFIG=s3_t3_nostage2_100k.yml ablations/slurm/cache.sh)
ablations/submit.sh --after $J s3_t3_nostage2_100k.yml
```

### b23cm1019
```bash
ablations/submit.sh --partition btech --time 12:00:00 --parts 2 \
  s3_s1_typeonly_100k.yml s3_k1_nocal_100k.yml s3_f2_middleonly_100k.yml s3_f4_randinit_100k.yml
```
That is 8 queued jobs. If the account's submitted-job limit refuses some, submit 1-2 runs now and
the rest later with `--after <last job id>`.

### akshatmishra (inside `screen -S abl`, env activated)
```bash
export SHARED_ROOT=/DATA/user/akshat_mishra1/krish/sevdcpt_shared OUT_ROOT=/DATA/user/akshat_mishra1/krish/sevdcpt
bash ablations/run_local.sh s3_k4_nokl_100k.yml s3_f1_noskip_100k.yml s3_s4_shared_100k.yml
```
Detach with `Ctrl-A D`, return with `screen -r abl`. If interrupted, run the same line again: each
run resumes from its newest checkpoint.

---

## 5. Check that a run started correctly

Logs: `ablations/logs/abl_<name>_p<part>_<jobid>.{out,err}` (Slurm) or
`$OUT_ROOT/ablations/logs/<config>.log` (akshatmishra). Look for:

- `preflight OK (fresh start)` (or `(resuming)` for a second part);
- `Visible GPU : ...` (Slurm only);
- `Number of train images: 13013`;
- iteration lines `[epoch ..., iter: ...] ... l_pix: ...`;
- on b23cm1019, part 2 shows `Resuming training from epoch: ..., iter: <multiple of 10000>`.

---

## 6. Where the checkpoints end up

`$OUT_ROOT/ablations/experiments/<name>_100k/models/net_g_100000.pth` (plus `net_g_latest.pth`),
with a checkpoint every 10k.

| Account | Getting checkpoints to the evaluation machine |
|---|---|
| p26cs0003 | local |
| b23cm1019 | p26cs0003 can read b23's scratch; confirm once with `ls` |
| akshatmishra | copy back only `net_g_100000.pth` (~550 MB each), or evaluate there |

Intermediate checkpoints (10k-90k) can be deleted after evaluation to save space.

---

## 7. After training: evaluation and decision (prepared separately)

Each model is evaluated on the CDD-11 test set, the severity-graded set (levels 1-3 and beyond
range), unseen degradation combinations, and analyzer-wrong / high-uncertainty images, then compared
with Plan B and DCPT at 100k using per-image paired tests. Decision rule, fixed in advance: a
component stays if it improves its target scenario by at least about 0.1 dB with a paired test
p < 0.01 and does not hurt elsewhere; otherwise it is removed.

| Date | Milestone |
|---|---|
| 6 Nov | priority-1 runs (C2, S1, K1, K2) trained |
| 7 Nov | priority-2 runs trained; evaluations start |
| 8 Nov | all evaluations done; decide which components stay |
| 9-10 Nov | abstract written around the final architecture |
| after 10 Nov | decisive variants re-run at 200k for the paper |
