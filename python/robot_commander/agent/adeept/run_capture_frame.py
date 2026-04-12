import argparse
from pathlib import Path

import cv2
from picamera2 import Picamera2


def main():
    parser = argparse.ArgumentParser(description="Capture a single camera frame to disk.")
    parser.add_argument("output", type=Path, nargs="?", default=Path("frame.jpg"), help="Output file path (default: frame.jpg)")
    args = parser.parse_args()

    camera = Picamera2()
    config = camera.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
    camera.configure(config)
    camera.start()

    frame = camera.capture_array()
    camera.stop()

    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(args.output), bgr_frame)
    print(f"Saved frame to {args.output}")


if __name__ == "__main__":
    main()
