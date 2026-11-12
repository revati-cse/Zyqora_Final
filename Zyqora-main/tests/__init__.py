"""Unit tests. Run from the project root:  python -m unittest discover -s tests -t . -v"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
