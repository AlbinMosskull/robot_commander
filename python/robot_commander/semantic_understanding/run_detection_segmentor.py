import cv2
import numpy as np

from robot_commander.image_processing.camera import WebCamera
from robot_commander.semantic_understanding.detection_segmentor import DetectionSegmentor
from robot_commander.semantic_understanding.semantic_types import SegmentationResult

_PALETTE = [
    (0, 100, 255),
    (0, 210, 0),
    (255, 80, 0),
    (255, 0, 180),
    (0, 220, 220),
    (200, 150, 0),
    (120, 0, 255),
    (0, 160, 80),
]



def _draw_results(frame: np.ndarray, results: list[SegmentationResult]) -> np.ndarray:
    overlay = frame.copy()
    for i, res in enumerate(reversed(results)):
        colour = _PALETTE[i % len(_PALETTE)]
        overlay[res.mask] = colour

    out = cv2.addWeighted(frame, 0.4, overlay, 0.6, 0)

    for i, res in enumerate(results):
        colour = _PALETTE[(len(results) - 1 - i) % len(_PALETTE)]
        ys, xs = np.where(res.mask)
        if len(xs) == 0:
            continue
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())

        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)

        label = f"{res.label} {res.score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), colour, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    return out


def main():
    print("Loading models...")
    segmentor = DetectionSegmentor()
    print("Models loaded. Press 'q' to quit.")

    with WebCamera() as cam:
        while True:
            ok, frame = cam.read()
            if not ok:
                print("Failed to read frame.")
                break

            sam_results = segmentor.process(frame)
            vis = _draw_results(frame, sam_results)

            label_counts: dict[str, int] = {}
            for res in sam_results:
                label_counts[res.label] = label_counts.get(res.label, 0) + 1
            summary = "  ".join(f"{v}x {k}" for k, v in sorted(label_counts.items()))
            if summary:
                cv2.putText(
                    vis, summary, (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                )

            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            small_vis = cv2.resize(vis, (0, 0), fx=0.5, fy=0.5)
            cv2.imshow("Camera", small_frame)
            cv2.imshow("Detections", small_vis)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
