import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (ROOT, os.path.join(ROOT, "target")):
    if p not in sys.path:
        sys.path.insert(0, p)

# Tests check the base configs; an experiment overlay exported in the shell must not leak in.
os.environ.pop("KAL_CONFIG_OVERLAY", None)
