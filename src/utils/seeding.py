from __future__ import annotations

import random
import time
from typing import Dict

import numpy as np


def seed_everything(seed: int | None = None) -> Dict[str, int]:
    """Inicializa los generadores de números aleatorios (random y NumPy).

    Si no se pasa seed, se genera una a partir del tiempo actual en milisegundos.
    Devuelve un diccionario con las seeds aplicadas para registrarlas en metadatos.
    """

    used_seed = int(seed) if seed is not None else int(time.time() * 1000) % (2**32)
    random.seed(used_seed)
    np.random.seed(used_seed)

    return {"python_random": used_seed, "numpy": used_seed}
