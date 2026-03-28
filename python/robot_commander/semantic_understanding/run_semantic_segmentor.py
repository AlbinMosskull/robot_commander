"""
Script to visualise instance segmentation on live camera frames.
"""

import cv2
import numpy as np

from robot_commander.image_processing.camera import Camera
from robot_commander.semantic_understanding.semantic_segmentor import (
    SemanticSegmentor,
    SegmentationResult,
)

# Palette of BGR colours cycled across instances.
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


def _draw_instances(frame: np.ndarray, instances: list[SegmentationResult]) -> np.ndarray:
    """Return a copy of *frame* with coloured instance masks and labels."""
    overlay = frame.copy()

    for i, inst in enumerate(reversed(instances)):
        colour = _PALETTE[i % len(_PALETTE)]
        overlay[inst.mask] = colour

    result = cv2.addWeighted(frame, 0.4, overlay, 0.6, 0)

    for i, inst in enumerate(instances):
        colour = _PALETTE[(len(instances) - 1 - i) % len(_PALETTE)]
        ys, xs = np.where(inst.mask)
        if len(xs) == 0:
            continue
        cx, cy = int(xs.mean()), int(ys.mean())
        label = f"{inst.label} {inst.score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(result, (cx - 2, cy - th - 4), (cx + tw + 2, cy + 2), (0, 0, 0), -1)
        cv2.putText(result, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1)

    return result


def main():
    print("Loading Mask2Former instance segmentation model...")
    segmentor = SemanticSegmentor()
    print(f"Model loaded. Press 'q' to quit.")

    with Camera() as cam:
        while True:
            ok, frame = cam.read()
            if not ok:
                print("Failed to read frame.")
                break

            instances = segmentor.process(frame)
            vis = _draw_instances(frame, instances)

            label_counts: dict[str, int] = {}
            for inst in instances:
                label_counts[inst.label] = label_counts.get(inst.label, 0) + 1
            summary = "  ".join(f"{v}x {k}" for k, v in sorted(label_counts.items()))
            if summary:
                cv2.putText(
                    vis, summary, (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                )

            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            small_vis = cv2.resize(vis, (0, 0), fx=0.5, fy=0.5)
            cv2.imshow("Camera", small_frame)
            cv2.imshow("Instances", small_vis)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
