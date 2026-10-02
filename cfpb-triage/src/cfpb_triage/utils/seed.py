"""Global seeding helper (Sayon)."""
import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    """Seed Python's random, NumPy's global RNG and PYTHONHASHSEED.

    sklearn estimators still need random_state=seed passed explicitly;
    PYTHONHASHSEED only affects subprocesses started after this call.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
