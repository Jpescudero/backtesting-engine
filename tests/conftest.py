from __future__ import annotations

import sys
from pathlib import Path

# Permite importar el paquete backtesting sin instalación previa
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
