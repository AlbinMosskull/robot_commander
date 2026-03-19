import numpy as np
import pytest
import torch

from robot_commander.depth_processing.depth_processor import DepthProcessor


@pytest.fixture(scope="module")
def processor():
    if not torch.cuda.is_available():
        pytest.skip("No CUDA GPU available")
    return DepthProcessor("depth-anything/Depth-Anything-V2-Small-hf")


def _make_frame(height: int = 480, width: int = 640) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (height, width, 3), dtype=np.uint8)


def test_output_is_2d(processor):
    result = processor.process(_make_frame())
    assert result.ndim == 2


def test_gives_expected_result_for_basic_frame(processor):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[240:, :] = 255  # Half white, half black
    result = processor.process(frame)
    assert result.shape == (480, 640)
    assert np.all(result[:240, :] < result[240:, :]), "Depth should be greater in the part of the frame that is white"
