import sys
import os
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv

MODEL_PATH = "runs/detect/runs/detect/runs/v2_nano_cont/weights/best.pt"
clip_name = sys.argv[1]
SOURCE_VIDEO = f"videos/soccernet_{clip_name}.mp4"
OUTPUT_VIDEO = f"runs/track/tracked_{clip_name}.mp4"

CLASS_NAMES = {0: "Player", 1: "Goalkeeper", 2: "Referee", 3: "Ball"}
BALL_CLASS = 3
MAX_MISSING_FRAMES = 45  # how many frames to "coast" the ball using predicted position

os.makedirs("runs/track", exist_ok=True)

model = YOLO(MODEL_PATH)
tracker = sv.ByteTrack()
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

cap = cv2.VideoCapture(SOURCE_VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

last_ball_pos = None      # (cx, cy, box_w, box_h)
last_ball_velocity = (0, 0)
missing_count = 0

frame_count = 0
for result in model.predict(source=SOURCE_VIDEO, stream=True, conf=0.5, imgsz=640):
    frame = result.orig_img
    detections = sv.Detections.from_ultralytics(result)
    detections = tracker.update_with_detections(detections)

    annotated = frame.copy()

    ball_found_this_frame = False
    for i in range(len(detections.xyxy)):
        cls = detections.class_id[i]
        if cls == BALL_CLASS:
            x1, y1, x2, y2 = detections.xyxy[i]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            bw, bh = x2 - x1, y2 - y1
            if last_ball_pos is not None:
                last_ball_velocity = (cx - last_ball_pos[0], cy - last_ball_pos[1])
            last_ball_pos = (cx, cy, bw, bh)
            missing_count = 0
            ball_found_this_frame = True

    # Draw all normal detections (player/GK/ref, and real ball if found)
    labels = [
        f"#{tid} {CLASS_NAMES.get(cls, 'Obj')}"
        for cls, tid in zip(detections.class_id, detections.tracker_id)
    ]
    annotated = box_annotator.annotate(scene=annotated, detections=detections)
    annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

    # If ball wasn't found, draw a predicted position using last known velocity
    if not ball_found_this_frame and last_ball_pos is not None and missing_count < MAX_MISSING_FRAMES:
        missing_count += 1
        vx, vy = last_ball_velocity
        pred_cx = last_ball_pos[0] + vx * missing_count
        pred_cy = last_ball_pos[1] + vy * missing_count
        bw, bh = last_ball_pos[2], last_ball_pos[3]
        x1, y1 = int(pred_cx - bw/2), int(pred_cy - bh/2)
        x2, y2 = int(pred_cx + bw/2), int(pred_cy + bh/2)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, "Ball (predicted)", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    out.write(annotated)
    frame_count += 1
    if frame_count % 50 == 0:
        print(f"Processed {frame_count} frames")

out.release()
print("Done. Saved to", OUTPUT_VIDEO)