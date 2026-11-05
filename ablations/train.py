"""Training entry point for the ablations.

Registers the ablation classes, installs the severity trainer's sampler-aware data loaders
(basicsr/severity_train.py, unchanged), then runs the shared trainer. Launched by
ablations/slurm/train.sh exactly like basicsr/severity_train.py:

  python -m torch.distributed.run --nproc_per_node=1 ablations/train.py -opt <rendered.yml> \
      --launcher pytorch --auto_resume
"""
import sys
from os import path as osp

ROOT = osp.abspath(osp.join(osp.dirname(__file__), osp.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import ablations.register  # noqa: E402,F401
import basicsr.severity_train as severity_train  # noqa: E402  (patches the trainer's data-loader builders)
import basicsr.all_in_one_train as base  # noqa: E402
from ablations.ablib.samplers import HoldoutFixedMixedSourceBatchSampler  # noqa: E402

# Same sampler, but a Stage 2 source may miss settings its dataset explicitly holds out (X3).
severity_train.FixedMixedSourceBatchSampler = HoldoutFixedMixedSourceBatchSampler

if __name__ == "__main__":
    base.train_pipeline(ROOT)
