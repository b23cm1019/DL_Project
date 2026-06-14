import argparse
import os
import random
from collections import OrderedDict
from os import path as osp

import torch
import yaml

from basicsr.utils import set_random_seed
from basicsr.utils.dist_util import get_dist_info, init_dist, master_only


def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        tuple: yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def yaml_load(f):
    """Load yaml file or string.

    Args:
        f (str): File path or a python string.

    Returns:
        dict: Loaded dict.
    """
    if os.path.isfile(f):
        with open(f, "r") as f:
            return yaml.load(f, Loader=ordered_yaml()[0])
    else:
        return yaml.load(f, Loader=ordered_yaml()[0])


def dict2str(opt, indent_level=1):
    """dict to string for printing options.

    Args:
        opt (dict): Option dict.
        indent_level (int): Indent level. Default: 1.

    Return:
        (str): Option string for printing.
    """
    msg = "\n"
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += " " * (indent_level * 2) + k + ":["
            msg += dict2str(v, indent_level + 1)
            msg += " " * (indent_level * 2) + "]\n"
        else:
            msg += " " * (indent_level * 2) + k + ": " + str(v) + "\n"
    return msg


def _postprocess_yml_value(value):
    # None
    if value == "~" or value.lower() == "none":
        return None
    # bool
    if value.lower() == "true":
        return True
    elif value.lower() == "false":
        return False
    # !!float number
    if value.startswith("!!float"):
        return float(value.replace("!!float", ""))
    # number
    # int
    if value.isdigit():
        return int(value)

    # float (including scientific notation)
    try:
        return float(value)
    except ValueError:
        pass
    # list
    if value.startswith("["):
        return eval(value)
    # str
    return value


def _remap_cdd11_path(path):
    """Remap hardcoded CDD-11 paths to the cluster dataset root when available."""
    data_root = os.environ.get("DCPT_DATA_ROOT")
    if not data_root or not path:
        return path

    normalized = path.replace("\\", "/")
    prefixes = {
        "/data/datasets/CDD-11_train/CDD-11_train/": osp.join(
            data_root, "train", "CDD-11_train"
        ),
        "/data/datasets/CDD-11_test/": osp.join(data_root, "test"),
    }

    for prefix, target_root in prefixes.items():
        if normalized.startswith(prefix):
            suffix = normalized[len(prefix) :].lstrip("/")
            return osp.join(target_root, suffix)
    return path


def parse_options(root_path, is_train=True):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-opt", type=str, required=True, help="Path to option YAML file."
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--auto_resume", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--force_yml",
        nargs="+",
        default=None,
        help="Force to update yml files. Examples: train:ema_decay=0.999",
    )
    args = parser.parse_args()

    # parse yml to dict
    opt = yaml_load(args.opt)

    # distributed settings
    if args.launcher == "none":
        opt["dist"] = False
        print("Disable distributed.", flush=True)
    else:
        opt["dist"] = True
        if args.launcher == "slurm" and "dist_params" in opt:
            init_dist(args.launcher, **opt["dist_params"])
        else:
            init_dist(args.launcher)
    opt["rank"], opt["world_size"] = get_dist_info()

    # random seed
    seed = opt.get("manual_seed")
    if seed is None:
        seed = random.randint(1, 10000)
        opt["manual_seed"] = seed
    set_random_seed(seed + opt["rank"])

    # force to update yml options
    if args.force_yml is not None:
        for entry in args.force_yml:
            # now do not support creating new keys
            keys, value = entry.split("=")
            keys, value = keys.strip(), value.strip()
            value = _postprocess_yml_value(value)
            eval_str = "opt"
            for key in keys.split(":"):
                eval_str += f'["{key}"]'
            eval_str += "=value"
            # using exec function
            exec(eval_str)

    opt["auto_resume"] = args.auto_resume
    opt["is_train"] = is_train

    # debug setting
    if args.debug and not opt["name"].startswith("debug"):
        opt["name"] = "debug_" + opt["name"]

    if opt["num_gpu"] == "auto":
        opt["num_gpu"] = torch.cuda.device_count()

    # datasets
    for phase, dataset in opt["datasets"].items():
        # for multiple datasets, e.g., val_1, val_2; test_1, test_2
        phase = phase.split("_")[0]
        dataset["phase"] = phase
        if "scale" in opt:
            dataset["scale"] = opt["scale"]
        if dataset.get("dataroot_gt") is not None:
            dataset["dataroot_gt"] = _remap_cdd11_path(
                osp.expanduser(dataset["dataroot_gt"])
            )
        if dataset.get("dataroot_lq") is not None:
            dataset["dataroot_lq"] = _remap_cdd11_path(
                osp.expanduser(dataset["dataroot_lq"])
            )

    # paths
    expandable_path_tokens = (
        "resume_state",
        "pretrain_network",
        "experiments_root",
        "results_root",
        "models",
        "training_states",
        "log",
        "visualization",
        "tb_logger",
    )
    for key, val in opt["path"].items():
        if (
            val is not None
            and isinstance(val, str)
            and any(token in key for token in expandable_path_tokens)
        ):
            opt["path"][key] = osp.expanduser(val)

    if is_train:
        experiments_root = opt["path"].get("experiments_root") or osp.join(
            root_path, "experiments", opt["name"]
        )
        opt["path"]["experiments_root"] = experiments_root
        opt["path"]["models"] = opt["path"].get("models") or osp.join(
            experiments_root, "models"
        )
        opt["path"]["training_states"] = opt["path"].get("training_states") or osp.join(
            experiments_root, "training_states"
        )
        opt["path"]["log"] = opt["path"].get("log") or experiments_root
        opt["path"]["visualization"] = opt["path"].get("visualization") or osp.join(
            experiments_root, "visualization"
        )
        opt["path"]["tb_logger"] = opt["path"].get("tb_logger") or osp.join(
            root_path, "tb_logger", opt["name"]
        )

        # change some options for debug mode
        if "debug" in opt["name"]:
            if "val" in opt:
                opt["val"]["val_freq"] = 8
            opt["logger"]["print_freq"] = 1
            opt["logger"]["save_checkpoint_freq"] = 8
    else:  # test
        results_root = opt["path"].get("results_root") or osp.join(
            root_path, "results", opt["name"]
        )
        opt["path"]["results_root"] = results_root
        opt["path"]["log"] = opt["path"].get("log") or results_root
        opt["path"]["visualization"] = opt["path"].get("visualization") or osp.join(
            results_root, "visualization"
        )

    return opt, args


@master_only
def copy_opt_file(opt_file, experiments_root):
    # copy the yml file to the experiment root
    import sys
    import time
    from shutil import copyfile

    cmd = " ".join(sys.argv)
    filename = osp.join(experiments_root, osp.basename(opt_file))
    copyfile(opt_file, filename)

    with open(filename, "r+") as f:
        lines = f.readlines()
        lines.insert(0, f"# GENERATE TIME: {time.asctime()}\n# CMD:\n# {cmd}\n\n")
        f.seek(0)
        f.writelines(lines)
