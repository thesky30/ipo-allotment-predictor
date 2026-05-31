import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


@pytest.fixture(scope="session")
def modeling_data():
    """Full training matrix (ipo_offline_sample ⋈ new_factor_panel)."""
    from baseline_models import load_modeling_data
    return load_modeling_data()
