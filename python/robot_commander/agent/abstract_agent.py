from abc import ABC, abstractmethod

from robot_commander.agent.data_types import RangeReading


class AbstractAgent(ABC):
    @abstractmethod
    def SetWaypointList(self, waypoints: list[tuple[float, float]]) -> None: ...

    @abstractmethod
    def GetXandY(self) -> tuple[float, float]: ...

    @abstractmethod
    def ObservePosition(self, x: float, y: float, confidence: float) -> None: ...

    @abstractmethod
    def GetSensorReading(self) -> list[RangeReading]: ...

    @abstractmethod
    def GetCameraReading(self): ...
