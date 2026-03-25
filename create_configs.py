import os
import yaml

base_dir = "c:/Users/krish/Desktop/DL_Project/dcpt/options/cdd_experiments"
os.makedirs(base_dir, exist_ok=True)

# Define the 11 degradation names based on CDD
cdd_degradations = [
    "low", "haze", "rain", "snow", 
    "low_haze", "low_rain", "low_snow", "haze_rain", "haze_snow",
    "low_haze_rain", "low_haze_snow"
]

def make_datasets(phase):
    datasets = {}
    for i, deg in enumerate(cdd_degradations):
        key = f"{phase}_{i+1}"
        datasets[key] = {
            "name": deg,
            "type": "PairedImageDataset",
            "dataroot_gt": f"path/to/CDD/{phase}/clear",
            "dataroot_lq": f"path/to/CDD/{phase}/{deg}",
            "io_backend": {"type": "disk"},
        }
        if phase == "train":
            datasets[key]["gt_size"] = 128
            datasets[key]["use_hflip"] = True
            datasets[key]["use_rot"] = True
            datasets[key]["dataset_enlarge_ratio"] = 1
        else:
            datasets[key]["num_worker_per_gpu"] = 1
            datasets[key]["batch_size_per_gpu"] = 1
    return datasets

train_datasets = make_datasets("train")
val_datasets = make_datasets("val")
# Add val datasets to train config 
for k, v in val_datasets.items():
    train_datasets[k] = v

# Baseline Pretrain
baseline_pretrain = {
    "name": "Row_B_Baseline_Pretrain_CDD11",
    "model_type": "DCPTModel",
    "scale": 1,
    "num_gpu": 1,
    "manual_seed": 0,
    "classify": True,
    "datasets": train_datasets,
    "dataloader": {
        "phase": "train",
        "use_shuffle": True,
        "num_worker_per_gpu": 2,
        "batch_size_per_gpu": 8,
        "dataset_enlarge_ratio": 1,
        "pin_memory": True,
    },
    "dataloader_val": {
        "phase": "val",
        "use_shuffle": False,
        "num_worker_per_gpu": 1,
        "batch_size_per_gpu": 1,
        "pin_memory": True,
    },
    "network_g": {
        "type": "NAFNet",
        "width": 64,
        "enc_blk_nums": [1, 1, 1, 28],
        "middle_blk_num": 1,
        "dec_blk_nums": [1, 1, 1, 1],
        "window_size": 8,
    },
    "find_unused_parameters": False,
    "hook_names": "decoder",
    "network_dc": {
        "type": "PromptIR_NoImg_DC",
        "feature_dims": [64, 128, 256, 512],
        "num_classes": 11, # Baseline uses 11 classes for FocalLoss
    },
    "path": {
        "pretrain_network_g": None,
        "strict_load_g": False,
        "resume_state": None,
    },
    "train": {
        "ema_decay": 0.999,
        "optim_g": {
            "type": "AdamW", "lr": 1e-4, "weight_decay": 0, "betas": [0.9, 0.99]
        },
        "optim_dc": {
            "type": "AdamW", "lr": 1e-4, "weight_decay": 1e-4, "betas": [0.9, 0.99]
        },
        "scheduler": {
            "type": "CosineAnnealingRestartLR",
            "periods": [25000],
            "restart_weights": [1],
            "eta_min": 1e-6,
        },
        "total_iter": 25000,
        "warmup_iter": -1,
        "pixel_opt": {"type": "L1Loss", "loss_weight": 1.0, "reduction": "mean"},
        "classify_opt": {"type": "FocalLoss", "gamma": 2.0},
    },
    "val": {
        "val_freq": 5000.0,
        "save_img": False,
        "metrics": {
            "psnr": {"type": "calculate_psnr", "crop_border": 0, "test_y_channel": True, "image_range": 255.0},
            "ssim": {"type": "calculate_ssim", "crop_border": 0, "test_y_channel": True, "image_range": 255.0},
        }
    },
    "logger": {"print_freq": 100, "save_checkpoint_freq": 5000.0, "use_tb_logger": True, "wandb": {"project": None}},
}

# Multi-label Pretrain
multilabel_pretrain = baseline_pretrain.copy()
multilabel_pretrain["name"] = "Row_C_MultiLabel_Pretrain_CDD11"
multilabel_pretrain["network_dc"] = multilabel_pretrain["network_dc"].copy()
multilabel_pretrain["network_dc"]["num_classes"] = 4 # Multi-label predicts 4 primitives
multilabel_pretrain["train"] = multilabel_pretrain["train"].copy()
multilabel_pretrain["train"]["classify_opt"] = {"type": "MultiLabelBCELoss"}

# Finetune helper
def create_finetune(name, model_type, pretrain_g, pretrain_dc=None, prompt_dim=0):
    cfg = multilabel_pretrain.copy()
    cfg["name"] = name
    cfg["model_type"] = model_type
    cfg["network_g"] = cfg["network_g"].copy()
    if prompt_dim > 0:
        cfg["network_g"]["prompt_dim"] = prompt_dim
    
    cfg["path"] = {"pretrain_network_g": pretrain_g, "strict_load_g": True}
    if pretrain_dc:
        cfg["path"]["pretrain_network_dc"] = pretrain_dc
        cfg["path"]["strict_load_dc"] = True
    
    cfg["train"] = cfg["train"].copy()
    cfg["train"]["optim_g"]["lr"] = 5e-5
    cfg["train"]["scheduler"]["periods"] = [100000]
    cfg["train"]["total_iter"] = 100000
    if "optim_dc" in cfg["train"]:
        del cfg["train"]["optim_dc"]
    if "classify_opt" in cfg["train"]:
        del cfg["train"]["classify_opt"]
    return cfg

baseline_finetune = create_finetune("Row_B_Baseline_Finetune_CDD11", "SRModel", "experiments/Row_B_Baseline_Pretrain_CDD11/models/net_g_latest.pth")
multilabel_finetune = create_finetune("Row_C_MultiLabel_Finetune_CDD11", "SRModel", "experiments/Row_C_MultiLabel_Pretrain_CDD11/models/net_g_latest.pth")

# Prompt Finetune (Row D) - uses PromptSRModel and loads frozen DC
prompt_finetune = create_finetune("Row_D_Prompt_Finetune_CDD11", "PromptSRModel", "experiments/Row_C_MultiLabel_Pretrain_CDD11/models/net_g_latest.pth", "experiments/Row_C_MultiLabel_Pretrain_CDD11/models/net_dc_latest.pth", prompt_dim=4)

# Test helper
def create_test(name, pretrain_g, prompt_dim=0):
    cfg = {
        "name": name,
        "model_type": "SRModel" if prompt_dim == 0 else "PromptSRModel",
        "scale": 1,
        "num_gpu": 1,
        "manual_seed": 0,
        "datasets": make_datasets("test"),
        "network_g": {
            "type": "NAFNet",
            "width": 64,
            "enc_blk_nums": [1, 1, 1, 28],
            "middle_blk_num": 1,
            "dec_blk_nums": [1, 1, 1, 1],
            "window_size": 16,
        },
        "path": {"pretrain_network_g": pretrain_g, "param_key_g": "params_ema", "strict_load_g": True},
        "val": {
            "save_img": True,
            "metrics": {
                "psnr": {"type": "calculate_psnr", "crop_border": 0, "test_y_channel": False, "image_range": 1.0},
                "ssim": {"type": "calculate_ssim", "crop_border": 0, "test_y_channel": False, "image_range": 1.0},
            }
        }
    }
    if prompt_dim > 0:
        cfg["network_g"]["prompt_dim"] = prompt_dim
        cfg["network_dc"] = {
            "type": "PromptIR_NoImg_DC",
            "feature_dims": [64, 128, 256, 512],
            "num_classes": 4,
        }
        cfg["hook_names"] = "decoder"
        cfg["path"]["pretrain_network_dc"] = "experiments/Row_C_MultiLabel_Pretrain_CDD11/models/net_dc_latest.pth"
        cfg["path"]["strict_load_dc"] = True
    return cfg

test_baseline = create_test("Test_Row_B", "experiments/Row_B_Baseline_Finetune_CDD11/models/net_g_latest.pth")
test_multilabel = create_test("Test_Row_C", "experiments/Row_C_MultiLabel_Finetune_CDD11/models/net_g_latest.pth")
test_prompt = create_test("Test_Row_D", "experiments/Row_D_Prompt_Finetune_CDD11/models/net_g_latest.pth", prompt_dim=4)

# Create scripts
# Pretrain baseline
with open(os.path.join(base_dir, "pretrain_baseline.yml"), "w") as f:
    yaml.dump(baseline_pretrain, f, sort_keys=False)
with open(os.path.join(base_dir, "pretrain_multilabel.yml"), "w") as f:
    yaml.dump(multilabel_pretrain, f, sort_keys=False)
with open(os.path.join(base_dir, "finetune_baseline.yml"), "w") as f:
    yaml.dump(baseline_finetune, f, sort_keys=False)
with open(os.path.join(base_dir, "finetune_multilabel.yml"), "w") as f:
    yaml.dump(multilabel_finetune, f, sort_keys=False)
with open(os.path.join(base_dir, "finetune_prompt.yml"), "w") as f:
    yaml.dump(prompt_finetune, f, sort_keys=False)
with open(os.path.join(base_dir, "test_baseline.yml"), "w") as f:
    yaml.dump(test_baseline, f, sort_keys=False)
with open(os.path.join(base_dir, "test_multilabel.yml"), "w") as f:
    yaml.dump(test_multilabel, f, sort_keys=False)
with open(os.path.join(base_dir, "test_prompt.yml"), "w") as f:
    yaml.dump(test_prompt, f, sort_keys=False)

print("All config files generated successfully at", base_dir)
