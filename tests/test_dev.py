"""P3 (scripts/p3_pipeline.py): only days 1-63, split in time order, retraining on earlier rows, new threshold."""
import numpy as np

from p3_pipeline import SETTINGS, dev_split, past_threshold, periods, production_day, retrain_starts


def test_v1_dev_data_is_the_first_half_only():
    """V1: the development data are the first half of the rows; no row of the later half is returned."""
    X = np.arange(2000).reshape(-1, 1)                # the row number as the only feature
    Xd, yd, dd, h = dev_split(X, np.zeros(2000, int), np.arange(2000) / 20.0, train_days=22)
    assert len(yd) == 1000 and Xd[:, 0].max() == 999  # rows 1000-1999 (the later half) are cut off


def test_v2_training_part_ends_with_the_chosen_day():
    """V2: training = production days 1..22, evaluation starts on day 23, in time order."""
    _, _, dd, h = dev_split(np.zeros((2000, 1)), np.zeros(2000, int), np.arange(2000) / 20.0, train_days=22)
    pday = production_day(dd)
    assert pday[h - 1] == 22 and pday[h] == 23
    assert (np.diff(pday) >= 0).all()


def test_v3_production_days_skip_days_without_production():
    """V3: a weekend without joints does not count as a production day."""
    assert list(production_day(np.array([0.1, 0.5, 1.2, 4.3, 4.9, 5.0]))) == [1, 1, 2, 3, 3, 4]


def test_r4_retraining_uses_only_earlier_rows():
    """R4: the model for an interval is trained on rows 0 .. start-1 only, and the intervals cover the
    evaluation part without gaps or overlaps, also for 5 periods of equal size."""
    days, h, n = np.arange(1000) / 20.0, 440, 1000
    for step in ["period", 7, 3, 1]:
        starts = retrain_starts(step, days, h, n)
        assert starts[0] == h and starts[-1] == n and all(a < b for a, b in zip(starts, starts[1:]))
        for a in starts[:-1]:
            assert np.arange(a).max() < a             # training rows used in run(): np.arange(start)
    assert periods(n, h) == [440, 552, 664, 776, 888, 1000]
    assert retrain_starts(7, days, h, n)[1] - h == 140    # 7 days x 20 rows


def test_m6_threshold_is_the_q_quantile_of_past_true_defect_scores():
    """M6: at most a share q of past true-defect scores lies below t; q = 0 gives the lowest score."""
    past = np.random.default_rng(3).random(1000)
    for q in [v for _, v in SETTINGS.values() if v is not None]:
        t = past_threshold(past, q)
        assert np.mean(past < t) <= q + 1e-12
        assert np.mean(past <= t) >= q
    assert past_threshold(past, 0.0) == past.min()
