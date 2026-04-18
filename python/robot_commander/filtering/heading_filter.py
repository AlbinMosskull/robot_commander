import math


def _normalize_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class HeadingFilter:
    def __init__(
        self,
        initial_heading: float,
        process_noise: float,
        measurement_noise: float,
        max_innovation: float = math.pi / 4,
    ):
        self._heading = initial_heading
        self._variance = 1.0
        self._process_noise = process_noise
        self._measurement_noise = measurement_noise
        self._max_innovation = max_innovation

    @property
    def heading(self) -> float:
        return self._heading

    @property
    def variance(self) -> float:
        return self._variance

    def predict(self, delta_heading: float) -> None:
        self._heading = _normalize_angle(self._heading + delta_heading)
        self._variance += self._process_noise

    def update(self, observed_heading: float) -> float | None:
        innovation = _normalize_angle(observed_heading - self._heading)
        if abs(innovation) > self._max_innovation and self._variance < self._max_innovation:
            return None
        gain = self._variance / (self._variance + self._measurement_noise)
        self._heading = _normalize_angle(self._heading + gain * innovation)
        self._variance = (1.0 - gain) * self._variance
        return innovation
