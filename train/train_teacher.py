"""Teacher (spec 7; CLAUDE.md sensor set rule 5): RGB + LWIR uncooled + LWIR cooled + MWIR. The cooled LWIR and
MWIR cameras are privileged, training-only information (SWIR deferred).

    python train/train_teacher.py --data out/dataset
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
from train.engine import cli  # noqa: E402

if __name__ == "__main__":
    cli(default_model="teacher")
