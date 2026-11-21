"""Data checks D1-D3 of docs/test_plan.md. They run on the real data after scripts/download_data.py
and are skipped where the data are not present (for example in CI, because of the licence)."""
from pathlib import Path

import pandas as pd
import pytest

DATA = Path(__file__).resolve().parents[1] / "data" / "dataset.csv"
pytestmark = pytest.mark.skipif(not DATA.exists(), reason="real data not downloaded")


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(DATA, index_col=0)


def test_d1_labels_are_0_or_1(df):
    assert set(df["class"].unique()) == {0, 1}


def test_d2_no_missing_values(df):
    assert df.drop(columns="timestamp").isna().sum().sum() == 0


def test_d3_records_in_time_order(df):
    assert pd.to_datetime(df["timestamp"]).is_monotonic_increasing
