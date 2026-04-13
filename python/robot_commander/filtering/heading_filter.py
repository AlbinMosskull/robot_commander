import math


def _normalize_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class HeadingFilter:
    def __init__(self, initial_heading: float, process_noise: float, measurement_noise: float):
        self._heading = initial_heading
        self._variance = 1.0
        self._process_noise = process_noise
        self._measurement_noise = measurement_noise

    @property
    def heading(self) -> float:
        return self._heading

    def predict(self, delta_heading: float) -> None:
        self._heading = _normalize_angle(self._heading + delta_heading)
        self._variance += self._process_noise

    def update(self, observed_heading: float) -> None:
        innovation = _normalize_angle(observed_heading - self._heading)
        gain = self._variance / (self._variance + self._measurement_noise)
        self._heading = _normalize_angle(self._heading + gain * innovation)
        self._variance = (1.0 - gain) * self._variance
