import numpy as np
import pytest

from robot_commander.filtering.kalman_filter import KalmanFilter


def _make_1d_filter(q: float = 0.01, r: float = 1.0, x0: float = 0.0) -> KalmanFilter:
    return KalmanFilter(
        F=np.array([[1.0]]),
        B=np.array([[0.0]]),
        H=np.array([[1.0]]),
        Q=np.array([[q]]),
        R=np.array([[r]]),
        x0=np.array([[x0]]),
        P0=np.array([[1.0]]),
    )


def test_predict_increases_covariance():
    kf = _make_1d_filter()
    p_before = kf.P.copy()
    kf.predict(np.array([[0.0]]))
    assert kf.P[0, 0] > p_before[0, 0]


def test_update_pulls_estimate_toward_measurement():
    kf = _make_1d_filter(x0=0.0)
    kf.predict(np.array([[0.0]]))
    kf.update(np.array([[5.0]]))
    assert 0.0 < kf.x[0, 0] < 5.0


def test_converges_to_constant_signal():
    true_value = 3.0
    kf = _make_1d_filter(q=0.01, r=0.5, x0=0.0)
    rng = np.random.default_rng(seed=0)

    for _ in range(100):
        kf.predict(np.array([[0.0]]))
        noisy_measurement = true_value + rng.normal(scale=0.5)
        kf.update(np.array([[noisy_measurement]]))

    assert abs(kf.x[0, 0] - true_value) < 0.2