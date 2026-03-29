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
    port: int


@dataclass(frozen=True)
class Config:
    camera: CameraConfig
    checkerboard: CheckerboardConfig
    tag: TagConfig
    agent: AgentConfig


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
            port=raw["agent"]["port"],
        ),
    )
