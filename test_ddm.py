from ddm import detect_ddm_drift
def test_ddm_no_drift():
    # Stream with a constant 10% error rate spread evenly (no drift)
    stream = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1] * 10
    assert detect_ddm_drift(stream) == -1

def test_ddm_sudden_drift():
    # 0% error rate followed by 100% error rate (obvious drift)
    stream = [0]*50 + [1]*50
    drift_index = detect_ddm_drift(stream)
    assert drift_index != -1
    # The drift should be detected after the error rate spikes
    assert drift_index >= 50

def test_ddm_min_instances():
    # 100% error rate, but too few instances to trigger (default is 30)
    stream = [1]*20
    assert detect_ddm_drift(stream) == -1