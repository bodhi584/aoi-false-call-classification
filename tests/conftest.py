"""Make the scripts importable in the tests and provide small made-up data (no real data in the repository)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


@pytest.fixture
def data():
    """2,000 made-up joints with 5 measurements; about 5 % true defects that score higher on feature 0."""
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.05).astype(int)
    X = rng.normal(size=(2000, 5))
    X[:, 0] += 2 * y
    return X, y


@pytest.fixture
def params():
    """Hyper-parameters in the format of the run log of Pfab & Rothering (2025)."""
    return {"n_estimators": 30, "max_features": 0, "class_weight": 2, "criterion": 0,
            "scaler_kind": 0, "sampling_ratio": 5.0, "SetThreshold": 0.5}
