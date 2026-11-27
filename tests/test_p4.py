"""Checks of the P4 threshold rules and period pooling (no data file needed)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from p4_sim import np_rank, periods, plain_rank, threshold  # noqa: E402


def test_np_rank_matches_p3_slides():
    assert [np_rank(m) for m in (298, 299, 500, 1000, 4622)] == [0, 1, 2, 5, 35]


def test_plain_rank_is_one_percent():
    assert [plain_rank(m) for m in (299, 500, 1000)] == [3, 6, 11]


def test_too_few_true_defects_release_nothing():
    assert threshold(np.linspace(0, 1, 100), "np") == -np.inf


def test_threshold_is_kth_lowest():
    s = np.arange(1000) / 1000
    assert threshold(s, "np") == s[4]          # k = 5 -> 5th lowest


def test_periods_pool_to_299():
    rec = [{"td": 200, "fc": 10, "missed": 1, "released": 5}, {"td": 150, "fc": 10, "missed": 0, "released": 5},
           {"td": 50, "fc": 10, "missed": 0, "released": 5}]
    p = periods(rec)
    assert len(p) == 1 and p[0]["td"] == 400
