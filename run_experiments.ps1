# 4-Day Plan: DCPT Research Project Execution Script (PowerShell)

# ==============================================================================
# GPU Settings
# ==============================================================================
$env:CUDA_VISIBLE_DEVICES="0"

# ==============================================================================
# Row B: Baseline (Single-Label FocalLoss)
# ==============================================================================
Write-Host "Starting Row B Pretraining (11-class single-label FocalLoss, 25k iters)" -ForegroundColor Cyan
# python -m torch.distributed.run --nproc_per_node=1 basicsr/all_in_one_train.py -opt options/cdd_experiments/pretrain_baseline.yml --launcher pytorch

Write-Host "Starting Row B Finetuning (100k iters)" -ForegroundColor Cyan
# python -m torch.distributed.run --nproc_per_node=1 basicsr/all_in_one_train.py -opt options/cdd_experiments/finetune_baseline.yml --launcher pytorch

# ==============================================================================
# Row C: Contribution 1 (Multi-Label BCE Loss)
# ==============================================================================
Write-Host "Starting Row C Pretraining (4-primitive multi-hot BCE, 25k iters)" -ForegroundColor Cyan
# python -m torch.distributed.run --nproc_per_node=1 basicsr/all_in_one_train.py -opt options/cdd_experiments/pretrain_multilabel.yml --launcher pytorch

Write-Host "Starting Row C Finetuning (100k iters)" -ForegroundColor Cyan
# python -m torch.distributed.run --nproc_per_node=1 basicsr/all_in_one_train.py -opt options/cdd_experiments/finetune_multilabel.yml --launcher pytorch

# ==============================================================================
# Row D: Contribution 2 (Row C + Semantic Soft Prompt Injection)
# ==============================================================================
Write-Host "Starting Row D Finetuning with Prompt Injection (uses Row C's Multilabel pretrain, 100k iters)" -ForegroundColor Cyan
# python -m torch.distributed.run --nproc_per_node=1 basicsr/all_in_one_train.py -opt options/cdd_experiments/finetune_prompt.yml --launcher pytorch

# ==============================================================================
# Testing
# ==============================================================================
Write-Host "Running Evaluation for Row B" -ForegroundColor Cyan
# python basicsr/test.py -opt options/cdd_experiments/test_baseline.yml

Write-Host "Running Evaluation for Row C" -ForegroundColor Cyan
# python basicsr/test.py -opt options/cdd_experiments/test_multilabel.yml

Write-Host "Running Evaluation for Row D" -ForegroundColor Cyan
# python basicsr/test.py -opt options/cdd_experiments/test_prompt.yml

Write-Host "All Experiments Concluded!" -ForegroundColor Green
