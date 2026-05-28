def _remap_cdd11_path(path):
    """Remap hardcoded CDD-11 paths to the cluster dataset root.

    Actual layout on cluster:
        Train: $DCPT_DATA_ROOT/train/CDD-11_train/<degradation>/
        Test:  $DCPT_DATA_ROOT/test/<degradation>/

    YML files hardcode:
        /data/datasets/CDD-11_train/CDD-11_train/<degradation>/
        /data/datasets/CDD-11_test/<degradation>/
    """
    data_root = os.environ.get("DCPT_DATA_ROOT")
    if not data_root or not path:
        return path

    normalized = path.replace("\\", "/")

    prefixes = {
        # train paths in yml → actual train location
        "/data/datasets/CDD-11_train/CDD-11_train/": os.path.join(
            data_root, "train", "CDD-11_train"
        ),
        # test paths in yml → actual test location
        "/data/datasets/CDD-11_test/": os.path.join(data_root, "test"),
    }

    for prefix, target_root in prefixes.items():
        if normalized.startswith(prefix):
            suffix = normalized[len(prefix):].lstrip("/")
            return os.path.join(target_root, suffix)
    return path
