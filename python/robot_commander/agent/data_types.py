from dataclasses import dataclass


@dataclass
class RangeReading:
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    did_hit: bool