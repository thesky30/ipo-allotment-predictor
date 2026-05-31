import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pandas as pd

from initial_data_analysis import excel_serial_to_datetime


def test_excel_serial_to_datetime_coerces_implausible_values_before_conversion():
    out = excel_serial_to_datetime(pd.Series([45500, 1e300, -1]))

    assert out.iloc[0] == pd.Timestamp("2024-07-27")
    assert pd.isna(out.iloc[1])
    assert pd.isna(out.iloc[2])


def test_excel_serial_to_datetime_handles_all_missing_series():
    out = excel_serial_to_datetime(pd.Series([None, float("nan")]))

    assert out.isna().all()
