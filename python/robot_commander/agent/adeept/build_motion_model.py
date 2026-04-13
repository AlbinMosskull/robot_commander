import math
import time

from robot_commander.image_processing import intrinsics as calibration
from robot_commander.image_processing.camera import WebCamera
from robot_commander.image_processing.tag_detector import TagDetector
from robot_commander.localization.localizer import Localizer
from robot_commander.localization.camera_localizer import CameraLocalizer
from robot_commander.localization.world_localizer import WorldPose
from robot_commander.map.map_coordinates import MapCoordinates
from robot_commander.remote_control.agent_client import AgentClient
from robot_commander.agent.adeept.adeept_motion_model import normalize_angle
from robot_commander.config import load as load_config

_cfg = load_config()
_SAMPLE_COUNT = 10
_SAMPLE_INTERVAL_S = 0.1
_FORWARD_DURATION_S = 3.0
_ROTATE_DURATION_S = 2.0
_SETTLE_S = 0.5


def _sample_pose(localizer: CameraLocalizer, cam: WebCamera, count: int = _SAMPLE_COUNT) -> WorldPose:
    poses = []
    while len(poses) < count:
        ok, frame = cam.read()
        if not ok:
            continue
        pose = localizer.localize(frame)
        if pose is not None:
            poses.append(pose)
        time.sleep(_SAMPLE_INTERVAL_S)
    x = sum(p.x for p in poses) / count
    y = sum(p.y for p in poses) / count
    heading = math.atan2(
        sum(math.sin(p.heading) for p in poses) / count,
        sum(math.cos(p.heading) for p in poses) / count,
    )
    return WorldPose(x, y, heading)


def main():
    cam_intrinsics = calibration.load()
    detector = TagDetector()
    localizer = Localizer(detector, cam_intrinsics.camera_matrix, _cfg.tag.size_m,
                          dist_coeffs=cam_intrinsics.dist_coeffs)
    map_coords = MapCoordinates.load(_cfg.map.stencil_path)
    camera_localizer = CameraLocalizer(localizer, map_coords)
    client = AgentClient()

    with WebCamera() as cam:
        cam.warm_up()

        print("Sampling initial position...")
        initial_pose = _sample_pose(camera_localizer, cam)
        print(f"  x={initial_pose.x:.3f}m  y={initial_pose.y:.3f}m  heading={math.degrees(initial_pose.heading):.1f}°")

        input("\nPress Enter to move forward...")
        client.run_command("forward", _FORWARD_DURATION_S)
        time.sleep(_SETTLE_S)

        print("Sampling position after forward move...")
        forward_pose = _sample_pose(camera_localizer, cam)
        print(f"  x={forward_pose.x:.3f}m  y={forward_pose.y:.3f}m  heading={math.degrees(forward_pose.heading):.1f}°")

        dx = forward_pose.x - initial_pose.x
        dy = forward_pose.y - initial_pose.y
        distance = math.hypot(dx, dy)
        forward_angle = math.atan2(dy, dx)

        heading_offset = normalize_angle(forward_angle - initial_pose.heading)

        print(f"\n--- Forward motion ---")
        print(f"  Distance:         {distance:.3f} m over {_FORWARD_DURATION_S}s")
        print(f"  V_FORWARD_M_S   = {distance / _FORWARD_DURATION_S:.3f}")
        print(f"  Forward angle:    {math.degrees(forward_angle):.1f}°  (initial heading: {math.degrees(initial_pose.heading):.1f}°)")
        print(f"  HEADING_OFFSET  = {math.degrees(heading_offset):.1f}°  ({heading_offset:.4f} rad)  (config: {_cfg.localization.heading_offset_deg:.1f}°)")

        input("\nPress Enter to start rotation calibration...")
        client.run_command("left", _ROTATE_DURATION_S)
        time.sleep(_SETTLE_S)

        print("Sampling heading after first rotation...")
        pose_1 = _sample_pose(camera_localizer, cam)
        print(f"  Heading 1: {math.degrees(pose_1.heading):.1f}°")

        client.run_command("left", _ROTATE_DURATION_S)
        time.sleep(_SETTLE_S)

        print("Sampling heading after second rotation...")
        pose_2 = _sample_pose(camera_localizer, cam)
        print(f"  Heading 2: {math.degrees(pose_2.heading):.1f}°")

        delta_heading = normalize_angle(pose_2.heading - pose_1.heading)
        omega = abs(delta_heading) / _ROTATE_DURATION_S

        print(f"\n--- Rotation ---")
        print(f"  Delta heading:    {math.degrees(delta_heading):.1f}°")
        print(f"  OMEGA_MAX_RAD_S = {omega:.3f}")

    client.close()


if __name__ == "__main__":
    main()
