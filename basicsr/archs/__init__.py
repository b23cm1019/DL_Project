import importlib
from copy import deepcopy
from os import path as osp

from basicsr.utils import get_root_logger, scandir
from basicsr.utils.registry import ARCH_REGISTRY

__all__ = ["build_network"]

# automatically scan and import arch modules for registry
# scan all the files under the 'archs' folder and collect files ending with '_arch.py'
arch_folder = osp.dirname(osp.abspath(__file__))
arch_filenames = [
    osp.splitext(osp.basename(v))[0]
    for v in scandir(arch_folder)
    if v.endswith("_arch.py")
]
_arch_import_errors = {}

for file_name in arch_filenames:
    module_name = f"basicsr.archs.{file_name}"
    try:
        importlib.import_module(module_name)
    except Exception as exc:
        _arch_import_errors[module_name] = exc


def _format_import_errors():
    if not _arch_import_errors:
        return ""
    lines = ["Deferred arch import errors:"]
    for module_name, exc in sorted(_arch_import_errors.items()):
        lines.append(f"  - {module_name}: {exc.__class__.__name__}: {exc}")
    return "\n".join(lines)


def build_network(opt):
    opt = deepcopy(opt)
    network_type = opt.pop("type")
    try:
        network_cls = ARCH_REGISTRY.get(network_type)
    except KeyError as exc:
        import_errors = _format_import_errors()
        if import_errors:
            raise KeyError(f"{exc}\n{import_errors}") from exc
        raise
    net = network_cls(**opt)
    logger = get_root_logger()
    logger.info(f"Network [{net.__class__.__name__}] is created.")
    return net
