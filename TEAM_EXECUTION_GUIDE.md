# DCPT Project: Team Execution Guide

This guide outlines exactly how to configure, run, and evaluate the DCPT research project modifications (Multi-Label BCE Loss and Semantic Soft Prompt Injection) on the CDD-11 Mixed Degradation Dataset. 

---

## Phase 1: Dataset Preparation

1.  **Download the CDD-11 Dataset**
    *   The CDD-11 dataset contains 11 permutations of 4 base primitive degradations (Low-light, Haze, Rain, Snow).
    *   Download links can be found on Kaggle (Search "CDD-11 Mixed Degradation Dataset") or HuggingFace (Search "OneRestore CDD-11").
2.  **Organize the Dataset Directory**
    *   Extract the dataset locally onto your machine. Ensure you have clear structures delineating `train` and `test` splits.
    *   For example:
        *   `C:/Datasets/CDD11/train/clear` (Ground Truth)
        *   `C:/Datasets/CDD11/train/low`
        *   `C:/Datasets/CDD11/train/haze_rain`
        *   `C:/Datasets/CDD11/test/clear`
        *   `C:/Datasets/CDD11/test/low_haze_snow`

---

## Phase 2: Configuration Modifications

Before running any code, you **must update the dataset paths** inside the configuration files to point to where you saved CDD-11 on your local machine.

1.  Open `options/cdd_experiments/pretrain_baseline.yml`
    *   Scroll to the `datasets` section.
    *   Change `dataroot_gt` to your **Ground Truth** train path (e.g., `C:/Datasets/CDD11/train/clear`).
    *   Change `dataroot_lq` to the respective **degraded** train path (e.g., `C:/Datasets/CDD11/train/low`).
    *   Repeat this for all 11 degradation dictionary entries in `train` and `val`.
2.  Repeat step 1 for the remaining `options/cdd_experiments/*.yml` files.
    *   *Tip:* Because `create_configs.py` generated these identically using placeholder paths, you can do a global "Find and Replace" in your code editor to swap `path/to/CDD/train` to your actual path across all 8 YAML files simultaneously.

---

## Phase 3: Running the Experiments

We have provided a unified script to run the PyTorch distributed training schedule. 
*   **For Windows via PowerShell:** Open `run_experiments.ps1`
*   **For Linux/Mac/Git Bash:** Open `run_experiments.sh`

### How to use the Script:
The script executes our experimental ablation study row-by-row. Because training takes time, the actual execution commands are **commented out** (using `#`).

**To run a step:**
1. Open the script file.
2. Remove the `#` in front of the specific command you want to run.
3. Save the file and run it in your terminal (e.g., `.\run_experiments.ps1`).
4. Wait for it to finish, then uncomment the next sequential step.

### Execution Order:

**Day 1: Row B (Baseline - 11 Single Classes)**
*   Run the **Row B Pretraining** command.
*   Run the **Row B Finetuning** command.

**Day 2: Row C (Contribution 1 - Multi-Label BCE)**
*   Run the **Row C Pretraining** command (Uses the `MultiLabelBCELoss`).
*   Run the **Row C Finetuning** command.

**Day 3: Row D (Contribution 2 - Semantic Soft Prompts)**
*   Run the **Row D Finetuning** command. *(Note: This utilizes the pretrained backbone generated from Row C, proving the transferability and prompt effectiveness).*

**Day 4: Evaluation & Testing**
*   Uncomment and run the **Testing** commands for Row B, Row C, and Row D successively.
*   The scripts will output PSNR and SSIM scores to terminal. Collect these metrics into your final ablation table to prove that `Row D > Row C > Row B`.

---

## Technical Context (For Team Review)

1.  **`basicsr/losses/classify_loss.py`**: Added `MultiLabelBCELoss`. It converts the incoming dataset index (0-10) to a 4-dimensional subset primitive (like `[1, 0, 1, 0]`) to train the model to recognize *multiple* simultaneous degradations using BCE loss.
2.  **`basicsr/archs/nafnet_arch.py`**: The `NAFNetBaseline` core forward pass was augmented. It now looks for an optional `prompt` tensor. If provided, it passes the 4-dimensional probability vector through lightweight linear layers (`prompt_proj`) and adds it deeply into the middle block of the NAFNet features.
3.  **`basicsr/models/prompt_sr_model.py`**: A brand new model wrapper created strictly for Finetuning Row D. It runs the input through NAFNet to generate intermediate features, uses the frozen Classifier to guess the multi-label degradation vector, and then runs NAFNet a second time to explicitly pass those predicted probabilities as a semantic soft prompt to guide the ultimate restoration.
