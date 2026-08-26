import sys
import os
from collections import defaultdict, deque

import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "runs/detect/runs/detect/runs/v2_nano_cont/weights/best.pt"

clip_name = sys.argv[1]

SOURCE_VIDEO = f"videos/soccernet_{clip_name}.mp4"
OUTPUT_VIDEO = f"runs/track/movement_paths_{clip_name}.mp4"

CLASS_NAMES = {
    0: "Player",
    1: "Goalkeeper",
    2: "Referee",
    3: "Ball"
}

PLAYER_CLASSES = {0, 1, 2}
BALL_CLASS = 3

# Number of previous positions kept for each tracked person.
# Higher = longer trail.
TRAIL_LENGTH = 40


# ============================================================
# PREPARE OUTPUT FOLDER
# ============================================================

os.makedirs("runs/track", exist_ok=True)


# ============================================================
# LOAD MODEL + TRACKER
# ============================================================

print("Loading YOLO model...")

model = YOLO(MODEL_PATH)

tracker = sv.ByteTrack()

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()


# ============================================================
# READ VIDEO INFORMATION
# ============================================================

cap = cv2.VideoCapture(SOURCE_VIDEO)

if not cap.isOpened():
    print(f"ERROR: Could not open video:")
    print(SOURCE_VIDEO)
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

cap.release()

if fps <= 0:
    fps = 30.0


# ============================================================
# OUTPUT VIDEO
# ============================================================

out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (w, h)
)


# ============================================================
# MOVEMENT HISTORY
#
# tracker_id -> list of (x, y) positions
#
# Each player/GK/referee gets their own history.
# ============================================================

track_history = defaultdict(lambda: deque(maxlen=TRAIL_LENGTH))


# ============================================================
# PROCESS VIDEO
# ============================================================

frame_count = 0

print()
print("Starting movement-path tracking...")
print(f"Input : {SOURCE_VIDEO}")
print(f"Output: {OUTPUT_VIDEO}")
print()


for result in model.predict(
    source=SOURCE_VIDEO,
    stream=True,
    conf=0.5,
    imgsz=640
):

    frame = result.orig_img
    detections = sv.Detections.from_ultralytics(result)

    # --------------------------------------------------------
    # UPDATE BYTE TRACK
    # --------------------------------------------------------

    detections = tracker.update_with_detections(detections)

    annotated = frame.copy()


    # --------------------------------------------------------
    # RECORD POSITIONS
    #
    # We use the bottom-center of the bounding box.
    #
    # This is a better representation of where a person is
    # standing/running on the ground than the box center.
    # --------------------------------------------------------

    for i in range(len(detections.xyxy)):

        cls = int(detections.class_id[i])

        # Ignore the ball for movement trails.
        if cls not in PLAYER_CLASSES:
            continue

        tracker_id = detections.tracker_id[i]

        # Safety check in case a detection has no ID.
        if tracker_id is None:
            continue

        tracker_id = int(tracker_id)

        x1, y1, x2, y2 = detections.xyxy[i]

        # Bottom-center point of person
        cx = int((x1 + x2) / 2)
        cy = int(y2)

        # Save this person's position.
        track_history[tracker_id].append((cx, cy))


    # --------------------------------------------------------
    # DRAW MOVEMENT TRAILS
    # --------------------------------------------------------

    for tracker_id, points in track_history.items():

        if len(points) < 2:
            continue

        points_list = list(points)

        for j in range(1, len(points_list)):

            pt1 = points_list[j - 1]
            pt2 = points_list[j]

            # Slightly fade older parts of the trail.
            age_ratio = j / len(points_list)

            thickness = max(
                1,
                int(2 * age_ratio)
            )

            cv2.line(
                annotated,
                pt1,
                pt2,
                (0, 255, 255),
                thickness,
                cv2.LINE_AA
            )


    # --------------------------------------------------------
    # LABEL DETECTIONS
    # --------------------------------------------------------

    labels = []

    for cls, tid in zip(
        detections.class_id,
        detections.tracker_id
    ):

        cls = int(cls)

        if tid is None:
            label = CLASS_NAMES.get(cls, "Object")
        else:
            label = f"#{int(tid)} {CLASS_NAMES.get(cls, 'Object')}"

        labels.append(label)


    # --------------------------------------------------------
    # DRAW DETECTION BOXES
    # --------------------------------------------------------

    annotated = box_annotator.annotate(
        scene=annotated,
        detections=detections
    )

    annotated = label_annotator.annotate(
        scene=annotated,
        detections=detections,
        labels=labels
    )


    # --------------------------------------------------------
    # WRITE FRAME
    # --------------------------------------------------------

    out.write(annotated)

    frame_count += 1

    if frame_count % 50 == 0:
        print(f"Processed {frame_count} frames")


# ============================================================
# FINISH
# ============================================================

out.release()

print()
print("==============================================")
print("Movement path processing complete.")
print("==============================================")
print(f"Frames processed : {frame_count}")
print(f"Output video     : {OUTPUT_VIDEO}")
print("==============================================")