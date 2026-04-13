from abc import ABC, abstractmethod

from robot_commander.sensor.range_reading import RangeReading


class AbstractAgent(ABC):
    @abstractmethod
    def SetWaypointList(self, waypoints: list[tuple[float, float]]) -> None: ...

    @abstractmethod
    def GetXandY(self) -> tuple[float, float]: ...

    @abstractmethod
    def ObservePosition(self, x: float, y: float, heading: float, confidence: float) -> None: ...

    @abstractmethod
    def GetSensorReading(self) -> list[RangeReading]: ...

    @abstractmethod
    def SetEscapePlan(self, waypoints: list[tuple[float, float]]) -> None: ...

    @abstractmethod
    def GetCameraReading(self): ...

    @abstractmethod
    def GetHeading(self) -> float: ...

    @abstractmethod
    def GetUltrasonicMin(self) -> float | None: ...

    @abstractmethod
    def RunCommand(self, command: str, duration_s: float) -> None: ...
