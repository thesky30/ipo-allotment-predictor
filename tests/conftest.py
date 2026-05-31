import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


@pytest.fixture(autouse=True)
def _numpy_product_errstate():
    """Run every test under the SAME numpy float-error mode the app uses
    (default: overflow/divide/invalid -> warn, not raise).

    Under pytest the global numpy error mode can get set to ``over='raise'``,
    which turns a benign precision-round overflow inside the reused training
    builders into an intermittent ``FloatingPointError`` (seen ~1/40 runs for
    one stock). The product runs under default mode and is unaffected; tests
    must mirror that. Saved/restored so we never leak state ourselves.
    """
    saved = np.geterr()
    np.seterr(over="warn", divide="warn", invalid="warn", under="ignore")
    try:
        yield
    finally:
        np.seterr(**saved)


@pytest.fixture(scope="session")
def modeling_data():
    """Full training matrix (ipo_offline_sample ⋈ new_factor_panel)."""
    from baseline_models import load_modeling_data
    return load_modeling_data()
