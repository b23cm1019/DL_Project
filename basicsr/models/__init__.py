import importlib
from copy import deepcopy
from os import path as osp

from basicsr.utils import get_root_logger, scandir
from basicsr.utils.registry import MODEL_REGISTRY

__all__ = ["build_model"]

# automatically scan and import model modules for registry
# scan all the files under the 'models' folder and collect files ending with '_model.py'
model_folder = osp.dirname(osp.abspath(__file__))
model_filenames = [
    osp.splitext(osp.basename(v))[0]
    for v in scandir(model_folder)
    if v.endswith("_model.py")
]
_model_import_errors = {}

for file_name in model_filenames:
    module_name = f"basicsr.models.{file_name}"
    try:
        importlib.import_module(module_name)
    except Exception as exc:
        _model_import_errors[module_name] = exc


def _format_import_errors():
    if not _model_import_errors:
        return ""
    lines = ["Deferred model import errors:"]
    for module_name, exc in sorted(_model_import_errors.items()):
        lines.append(f"  - {module_name}: {exc.__class__.__name__}: {exc}")
    return "\n".join(lines)


def build_model(opt):
    """Build model from options.

    Args:
        opt (dict): Configuration. It must contain:
            model_type (str): Model type.
    """
    opt = deepcopy(opt)
    try:
        model_cls = MODEL_REGISTRY.get(opt["model_type"])
    except KeyError as exc:
        import_errors = _format_import_errors()
        if import_errors:
            raise KeyError(f"{exc}\n{import_errors}") from exc
        raise
    model = model_cls(opt)
    logger = get_root_logger()
    logger.info(f"Model [{model.__class__.__name__}] is created.")
    return model
