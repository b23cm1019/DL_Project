import importlib
from copy import deepcopy
from os import path as osp

from basicsr.utils import get_root_logger, scandir
from basicsr.utils.registry import LOSS_REGISTRY

# from .gan_loss import g_path_regularize, gradient_penalty_loss, r1_penalty

# __all__ = ["build_loss", "gradient_penalty_loss", "r1_penalty", "g_path_regularize"]
__all__ = ["build_loss"]

# automatically scan and import loss modules for registry
# scan all the files under the 'losses' folder and collect files ending with '_loss.py'
loss_folder = osp.dirname(osp.abspath(__file__))
loss_filenames = [
    osp.splitext(osp.basename(v))[0]
    for v in scandir(loss_folder)
    if v.endswith("_loss.py")
]
_loss_import_errors = {}

for file_name in loss_filenames:
    module_name = f"basicsr.losses.{file_name}"
    try:
        importlib.import_module(module_name)
    except Exception as exc:
        _loss_import_errors[module_name] = exc


def _format_import_errors():
    if not _loss_import_errors:
        return ""
    lines = ["Deferred loss import errors:"]
    for module_name, exc in sorted(_loss_import_errors.items()):
        lines.append(f"  - {module_name}: {exc.__class__.__name__}: {exc}")
    return "\n".join(lines)


def build_loss(opt):
    """Build loss from options.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Model type.
    """
    opt = deepcopy(opt)
    loss_type = opt.pop("type")
    try:
        loss_cls = LOSS_REGISTRY.get(loss_type)
    except KeyError as exc:
        import_errors = _format_import_errors()
        if import_errors:
            raise KeyError(f"{exc}\n{import_errors}") from exc
        raise
    loss = loss_cls(**opt)
    logger = get_root_logger()
    logger.info(f"Loss [{loss.__class__.__name__}] is created.")
    return loss
