#!/bin/bash
# 4-Day Plan: DCPT Research Project Execution Script

# ==============================================================================
# GPU Settings
# ==============================================================================
export CUDA_VISIBLE_DEVICES=0

# ==============================================================================
# Row B: Baseline (Single-Label FocalLoss)
# ==============================================================================
echo "Starting Row B Pretraining (11-class single-label FocalLoss, 25k iters)"
torchrun --nproc_per_node=1 basicsr/all_in_one_train.py -opt options/cdd_experiments/pretrain_baseline.yml --launcher pytorch

echo "Starting Row B Finetuning (100k iters)"
torchrun --nproc_per_node=1 basicsr/all_in_one_train.py -opt options/cdd_experiments/finetune_baseline.yml --launcher pytorch

# ==============================================================================
# Row C: Contribution 1 (Multi-Label BCE Loss)
# ==============================================================================
echo "Starting Row C Pretraining (4-primitive multi-hot BCE, 25k iters)"
# torchrun --nproc_per_node=1 basicsr/all_in_one_train.py -opt options/cdd_experiments/pretrain_multilabel.yml --launcher pytorch

echo "Starting Row C Finetuning (100k iters)"
# torchrun --nproc_per_node=1 basicsr/all_in_one_train.py -opt options/cdd_experiments/finetune_multilabel.yml --launcher pytorch

# ==============================================================================
# Row D: Contribution 2 (Row C + Semantic Soft Prompt Injection)
# ==============================================================================
echo "Starting Row D Finetuning with Prompt Injection (uses Row C's Multilabel pretrain, 100k iters)"
# torchrun --nproc_per_node=1 basicsr/all_in_one_train.py -opt options/cdd_experiments/finetune_prompt.yml --launcher pytorch

# ==============================================================================
# Testing
# ==============================================================================
echo "Running Evaluation for Row B"
# python basicsr/test.py -opt options/cdd_experiments/test_baseline.yml

echo "Running Evaluation for Row C"
# python basicsr/test.py -opt options/cdd_experiments/test_multilabel.yml

echo "Running Evaluation for Row D"
# python basicsr/test.py -opt options/cdd_experiments/test_prompt.yml

echo "All Experiments Concluded!"
