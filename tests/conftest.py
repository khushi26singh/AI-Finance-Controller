"""
Shared pytest configuration.

Makes the sibling packages under src/ (data_generation, reconciliation,
llm_resolution, dashboard) importable as top-level modules from any test
file, without needing to install the project as a package.
"""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))