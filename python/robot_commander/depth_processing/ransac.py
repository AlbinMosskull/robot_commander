from dataclasses import dataclass

import numpy as np

from robot_commander.config import load as load_config

_CAMERA_UP = np.array(load_config().depth.camera_up)


@dataclass(frozen=True)
class Plane:
    """A plane detected in a point cloud.

    Attributes:
        normal: Unit normal vector, shape (3,).
        distance: Signed distance from the origin along the normal.
        inliers: Boolean mask of shape (N,) over the original input point cloud.
    """

    normal: np.ndarray
    distance: float
    inliers: np.ndarray

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Plane):
            return NotImplemented
        return (
            np.allclose(self.normal, other.normal)
            and np.isclose(self.distance, other.distance)
            and np.array_equal(self.inliers, other.inliers)
        )

    def __hash__(self) -> int:
        raise TypeError("Plane is not hashable")


def _fit_plane(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit a plane to a set of points using SVD.

    The normal is the right-singular vector corresponding to the smallest
    singular value — the axis of least variance.

    Returns:
        (normal, d): unit normal vector and signed distance from the origin.
    """
    centroid = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - centroid, full_matrices=False)
    normal = vh[-1] / np.linalg.norm(vh[-1])
    return normal, float(normal @ centroid)


def _point_plane_distances(points: np.ndarray, normal: np.ndarray, d: float) -> np.ndarray:
    """Return the absolute orthogonal distance of each point from the plane."""
    return np.abs(points @ normal - d)


def _is_collinear(sample: np.ndarray, tol: float = 1e-10) -> bool:
    cross = np.cross(sample[1] - sample[0], sample[2] - sample[0])
    return bool(np.linalg.norm(cross) < tol)


def _ransac_single_plane(
    points: np.ndarray,
    n_iterations: int,
    distance_threshold: float,
    min_inliers: int,
    rng: np.random.Generator,
) -> Plane | None:
    """Find the dominant plane in *points* via RANSAC.

    After the best consensus set is found the plane is refit using all inliers
    to produce a more accurate estimate.

    Returns:
        A Plane with an inlier mask over *points*, or None if no plane reached
        *min_inliers* support.
    """
    best_count = 0
    best_inliers: np.ndarray | None = None

    for _ in range(n_iterations):
        sample = points[rng.choice(len(points), 3, replace=False)]
        if _is_collinear(sample):
            continue

        normal, d = _fit_plane(sample)
        inliers = _point_plane_distances(points, normal, d) < distance_threshold
        count = int(inliers.sum())

        if count > best_count:
            best_count = count
            best_inliers = inliers

    if best_inliers is None or best_count < min_inliers:
        return None

    # Refit on the full consensus set for a more accurate normal and distance.
    normal, d = _fit_plane(points[best_inliers])
    refined_inliers = _point_plane_distances(points, normal, d) < distance_threshold

    return Plane(normal=normal, distance=d, inliers=refined_inliers)


def detect_planes(
    points: np.ndarray,
    n_planes: int = 3,
    n_iterations: int = 100,
    distance_threshold: float = 0.02,
    min_inlier_fraction: float = 0.05,
    seed: int | None = None,
) -> list[Plane]:
    """Detect planes in a point cloud using iterative RANSAC.

    Each iteration finds the dominant plane, removes its inliers, then repeats
    on the remaining points.  Inlier masks on returned planes always index into
    the original *points* array.

    Args:
        points: Input point cloud, shape (N, 3).
        n_planes: Maximum number of planes to extract.
        n_iterations: RANSAC iterations per plane.
        distance_threshold: Max point-to-plane distance to count as an inlier,
            in the same units as *points*.
        min_inlier_fraction: Fraction of the original point count below which a
            candidate plane is rejected.
        seed: Optional RNG seed for reproducibility.

    Returns:
        Planes ordered from most to fewest inliers.
    """
    rng = np.random.default_rng(seed)
    min_inliers = max(3, int(min_inlier_fraction * len(points)))

    remaining_idx = np.arange(len(points))
    planes: list[Plane] = []

    for _ in range(n_planes):
        if len(remaining_idx) < 3:
            break

        plane = _ransac_single_plane(
            points[remaining_idx], n_iterations, distance_threshold, min_inliers, rng
        )
        if plane is None:
            break

        global_inliers = np.zeros(len(points), dtype=bool)
        global_inliers[remaining_idx[plane.inliers]] = True

        planes.append(Plane(normal=plane.normal, distance=plane.distance, inliers=global_inliers))
        remaining_idx = remaining_idx[~plane.inliers]

    return planes


def detect_floor(
    points: np.ndarray,
    n_planes: int = 3,
    n_iterations: int = 100,
    distance_threshold: float = 0.03,
) -> Plane | None:
    planes = detect_planes(points, n_planes=n_planes, n_iterations=n_iterations, distance_threshold=distance_threshold)
    if not planes:
        return None
    floor = max(planes, key=lambda p: abs(float(p.normal @ _CAMERA_UP)))
    if floor.distance > 0:
        floor = Plane(-floor.normal, -floor.distance, floor.inliers)
    return floor
