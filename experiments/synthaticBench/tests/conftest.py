"""Test setup for the local SynthACticBench checkout."""

from __future__ import annotations

import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SYNTHACTIC_ROOT = REPOSITORY_ROOT / "external" / "SynthACticBench"

# This makes the tests work both with an editable installation and directly
# against the checkout in external/.
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(SYNTHACTIC_ROOT))
