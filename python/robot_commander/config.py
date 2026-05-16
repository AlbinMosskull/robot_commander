from dataclasses import dataclass
from pathlib import Path

import yaml

_CONFIG_PATH = Path(__file__).parent / "config.yaml"


@dataclass(frozen=True)
class CameraConfig:
    device_index: int
    width: int
    height: int
    preview_width: int
    preview_height: int
    exposure: int | None


@dataclass(frozen=True)
class CheckerboardConfig:
    cols: int
    rows: int
    square_size_m: float


@dataclass(frozen=True)
class TagConfig:
    size_m: float


@dataclass(frozen=True)
class AgentConfig:
    v_forward_m_s: float
    heading_offset_deg: float


@dataclass(frozen=True)
class ConnectionConfig:
    host: str
    port: int


@dataclass(frozen=True)
class MapConfig:
    stencil_path: Path


@dataclass(frozen=True)
class DepthConfig:
    cone_half_angle_deg: float
    camera_up: tuple[float, float, float]


@dataclass(frozen=True)
class Config:
    camera: CameraConfig
    checkerboard: CheckerboardConfig
    tag: TagConfig
    agent: AgentConfig
    connection: ConnectionConfig
    map: MapConfig
    depth: DepthConfig


def load(path: Path = _CONFIG_PATH) -> Config:
    raw = yaml.safe_load(path.read_text())
    cam = raw["camera"]
    cb = raw["checkerboard"]
    return Config(
        camera=CameraConfig(
            device_index=cam["device_index"],
            width=cam["width"],
            height=cam["height"],
            preview_width=cam["preview_width"],
            preview_height=cam["preview_height"],
            exposure=cam.get("exposure"),
        ),
        checkerboard=CheckerboardConfig(
            cols=cb["cols"],
            rows=cb["rows"],
            square_size_m=cb["square_size_m"],
        ),
        tag=TagConfig(
            size_m=raw["tag"]["size_m"],
        ),
        agent=AgentConfig(
            v_forward_m_s=raw["agent"]["v_forward_m_s"],
            heading_offset_deg=raw["agent"]["heading_offset_deg"],
        ),
        connection=ConnectionConfig(
            host=raw["connection"]["host"],
            port=raw["connection"]["port"],
        ),
        map=MapConfig(
            stencil_path=Path(raw["map"]["stencil_path"]),
        ),
        depth=DepthConfig(
            cone_half_angle_deg=raw["depth"]["cone_half_angle_deg"],
            camera_up=tuple(raw["depth"]["camera_up"]),
        ),
    )
