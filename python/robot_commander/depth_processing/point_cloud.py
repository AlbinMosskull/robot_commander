import numpy as np

from robot_commander.camera.intrinsics import Intrinsics


def depth_image_to_point_cloud(
    depth: np.ndarray,
    intrinsics: Intrinsics,
) -> np.ndarray:
    """Back-project a depth image to a 3-D point cloud using the pinhole model.

    Pixels with depth <= 0 are excluded.

    Args:
        depth: Depth map (H, W) in any consistent unit.
        intrinsics: Camera intrinsics.

    Returns:
        Float32 array of shape (N, 3) — each row is an (X, Y, Z) point in
        camera space, where Z points away from the camera.
    """
    h, w = depth.shape
    uu, vv = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

    valid = depth > 0
    z = depth[valid].astype(np.float32)
    x = (uu[valid] - intrinsics.cx) * z / intrinsics.fx
    y = (vv[valid] - intrinsics.cy) * z / intrinsics.fy

    return np.stack([x, y, z], axis=-1)
