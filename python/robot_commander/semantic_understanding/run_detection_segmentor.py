"""
Script to visualise object detection + SAM segmentation on live camera frames.

For each frame:
  1. DETR detects objects and produces bounding boxes.
  2. Overlapping boxes of the same class are merged into one.
  3. SAM refines each merged box into a precise mask.
  4. The result is rendered as a coloured mask overlay with labels.

Press 'q' to quit.
"""

import cv2
import numpy as np

from robot_commander.image_processing.camera import Camera
from robot_commander.semantic_understanding.detection_segmentor import DetectionSegmentor
from robot_commander.semantic_understanding.sam_segmentor import SamSegmentor
from robot_commander.semantic_understanding.semantic_segmentor import SegmentationResult

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


def _merge_overlapping_boxes(
    detections: list[SegmentationResult],
) -> list[tuple[str, float, tuple[int, int, int, int]]]:
    """
    Group detections by label and merge any boxes that overlap.

    Two boxes are merged when they share any pixel area. Merging is repeated
    until no overlapping pair remains (handles transitive overlaps).

    Returns:
        List of (label, best_score, (x1, y1, x2, y2)) — one entry per merged group.
    """
    # Collect bboxes per label from the rectangular masks.
    by_label: dict[str, list[list]] = {}
    for det in detections:
        ys, xs = np.where(det.mask)
        if len(xs) == 0:
            continue
        box = [det.score, int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        by_label.setdefault(det.label, []).append(box)

    merged_prompts: list[tuple[str, float, tuple[int, int, int, int]]] = []
    for label, boxes in by_label.items():
        # Repeatedly scan for overlapping pairs and merge them.
        changed = True
        while changed:
            changed = False
            out: list[list] = []
            used = [False] * len(boxes)
            for i in range(len(boxes)):
                if used[i]:
                    continue
                s, ax1, ay1, ax2, ay2 = boxes[i]
                for j in range(i + 1, len(boxes)):
                    if used[j]:
                        continue
                    sj, bx1, by1, bx2, by2 = boxes[j]
                    if ax1 <= bx2 and bx1 <= ax2 and ay1 <= by2 and by1 <= ay2:
                        ax1, ay1 = min(ax1, bx1), min(ay1, by1)
                        ax2, ay2 = max(ax2, bx2), max(ay2, by2)
                        s = max(s, sj)
                        used[j] = True
                        changed = True
                out.append([s, ax1, ay1, ax2, ay2])
            boxes = out

        for s, x1, y1, x2, y2 in boxes:
            merged_prompts.append((label, s, (x1, y1, x2, y2)))

    return merged_prompts


def _draw_results(frame: np.ndarray, results: list[SegmentationResult]) -> np.ndarray:
    """Return a copy of *frame* with SAM mask overlays, bounding boxes, and labels."""
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
    print("Loading DETR object detection model...")
    detector = DetectionSegmentor()
    print("Loading SAM model...")
    sam = SamSegmentor()
    print("Models loaded. Press 'q' to quit.")

    with Camera() as cam:
        while True:
            ok, frame = cam.read()
            if not ok:
                print("Failed to read frame.")
                break

            detections = detector.process(frame)
            prompts = _merge_overlapping_boxes(detections)
            sam_results = sam.process(frame, prompts)
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
