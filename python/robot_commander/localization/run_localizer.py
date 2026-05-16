import math

import cv2

from robot_commander.image_processing import intrinsics as calibration
from robot_commander.image_processing.camera import WebCamera
from robot_commander.image_processing.tag_detector import TagDetector, draw_tags
from robot_commander.config import load as load_config
from robot_commander.localization.localizer import Localizer
from robot_commander.localization.camera_localizer import CameraLocalizer
from robot_commander.map.map_coordinates import MapCoordinates

_cfg = load_config()


def main():
    cam_intrinsics = calibration.load()
    detector = TagDetector()
    localizer = Localizer(detector, cam_intrinsics.camera_matrix, _cfg.tag.size_m,
                          dist_coeffs=cam_intrinsics.dist_coeffs)
    map_coords = MapCoordinates.load(_cfg.map.stencil_path)
    heading_offset_rad = math.radians(_cfg.agent.heading_offset_deg)
    camera_localizer = CameraLocalizer(localizer, map_coords, heading_offset=heading_offset_rad)

    with WebCamera() as cam:
        print("Camera opened. Press 'q' to quit.")
        cam.warm_up()

        while True:
            ok, frame = cam.read()
            if not ok:
                print("Failed to read frame.")
                break

            tags = detector.detect(frame)
            annotated = draw_tags(frame, tags)

            pose = camera_localizer.localize(frame)
            if pose is not None:
                heading_deg = math.degrees(pose.heading)
                label = f"x={pose.x:.3f}m  y={pose.y:.3f}m  heading={heading_deg:.1f}deg"
                print(f"  {label}")
                cv2.putText(annotated, label, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            else:
                cv2.putText(annotated, "No tag detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            display = cv2.resize(annotated, (_cfg.camera.preview_width, _cfg.camera.preview_height))
            cv2.imshow("Localization", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
