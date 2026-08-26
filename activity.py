from ultralytics import YOLO
import cv2
import numpy as np
import sys
import os


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = r"D:\FootballAI\Traning3\best.pt"

TEAM_A_COLOR = (255, 0, 0)       # Blue
TEAM_B_COLOR = (0, 0, 255)       # Red
GOALKEEPER_COLOR = (0, 255, 255) # Yellow
REFEREE_COLOR = (255, 0, 255)    # Purple
BALL_COLOR = (0, 255, 0)         # Green

CONFIDENCE = 0.25

# Movement thresholds in pixels/frame.
# These are deliberately conservative.
STANDING_THRESHOLD = 0.8
WALKING_THRESHOLD = 2.5
JOGGING_THRESHOLD = 5.0

# Number of frames used to smooth activity classification.
SMOOTHING_FRAMES = 7


# ============================================================
# GET VIDEO PATH
# ============================================================

if len(sys.argv) < 2:
    print("Usage:")
    print(
        r'python scripts\activity.py "D:\FootballAI\videos\sequences\SNMOT-140.mp4"'
    )
    sys.exit(1)

VIDEO_PATH = sys.argv[1]

if not os.path.exists(VIDEO_PATH):
    print(f"Video not found: {VIDEO_PATH}")
    sys.exit(1)


# ============================================================
# LOAD MODEL
# ============================================================

model = YOLO(MODEL_PATH)


# ============================================================
# OUTPUT
# ============================================================

output_dir = r"D:\FootballAI\outputs\activity"
os.makedirs(output_dir, exist_ok=True)

video_name = os.path.splitext(
    os.path.basename(VIDEO_PATH)
)[0]

output_path = os.path.join(
    output_dir,
    video_name + "_activity.mp4"
)


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 25.0

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    output_path,
    fourcc,
    fps,
    (width, height)
)


# ============================================================
# MOVEMENT MEMORY
# ============================================================

previous_positions = {}

activity_history = {}


# ============================================================
# TEAM ACTIVITY COUNTERS
# ============================================================

team_a_activity = {
    "STANDING": 0,
    "WALKING": 0,
    "JOGGING": 0,
    "RUNNING": 0
}

team_b_activity = {
    "STANDING": 0,
    "WALKING": 0,
    "JOGGING": 0,
    "RUNNING": 0
}


# ============================================================
# ACTIVITY CLASSIFIER
# ============================================================

def classify_activity(speed):

    if speed < STANDING_THRESHOLD:
        return "STANDING"

    elif speed < WALKING_THRESHOLD:
        return "WALKING"

    elif speed < JOGGING_THRESHOLD:
        return "JOGGING"

    else:
        return "RUNNING"


# ============================================================
# MAIN PROCESSING
# ============================================================

print("\nProcessing activity classification...")
print(f"Video: {os.path.basename(VIDEO_PATH)}")

frame_number = 0


while True:

    success, frame = cap.read()

    if not success:
        break

    frame_number += 1


    # ========================================================
    # TRACK PLAYERS
    # ========================================================

    results = model.track(
        frame,
        conf=CONFIDENCE,
        tracker="bytetrack.yaml",
        persist=True,
        verbose=False
    )

    result = results[0]


    # Current-frame activity counts
    current_team_a = {
        "STANDING": 0,
        "WALKING": 0,
        "JOGGING": 0,
        "RUNNING": 0
    }

    current_team_b = {
        "STANDING": 0,
        "WALKING": 0,
        "JOGGING": 0,
        "RUNNING": 0
    }


    # ========================================================
    # PROCESS DETECTIONS
    # ========================================================

    if result.boxes is not None:

        for box in result.boxes:

            cls_id = int(box.cls[0])

            class_name = model.names[cls_id]

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )


            # =================================================
            # PLAYER
            # =================================================

            if class_name == "Player":

                if box.id is None:
                    continue

                track_id = int(box.id[0])


                # ---------------------------------------------
                # PLAYER CENTER
                # ---------------------------------------------

                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                current_position = (
                    center_x,
                    center_y
                )


                # ---------------------------------------------
                # CALCULATE MOVEMENT
                # ---------------------------------------------

                if track_id in previous_positions:

                    previous_x, previous_y = previous_positions[
                        track_id
                    ]

                    pixel_distance = np.sqrt(
                        (center_x - previous_x) ** 2 +
                        (center_y - previous_y) ** 2
                    )

                else:

                    pixel_distance = 0.0


                previous_positions[track_id] = current_position


                # ---------------------------------------------
                # ACTIVITY
                # ---------------------------------------------

                raw_activity = classify_activity(
                    pixel_distance
                )


                # ---------------------------------------------
                # SMOOTH ACTIVITY
                # ---------------------------------------------

                if track_id not in activity_history:

                    activity_history[track_id] = []

                activity_history[track_id].append(
                    raw_activity
                )

                if len(activity_history[track_id]) > SMOOTHING_FRAMES:

                    activity_history[track_id].pop(0)


                history = activity_history[track_id]


                # Most common activity in recent frames
                activity = max(
                    set(history),
                    key=history.count
                )


                # =================================================
                # DETERMINE TEAM
                # =================================================

                box_width = x2 - x1
                box_height = y2 - y1

                if box_width <= 0 or box_height <= 0:
                    continue


                jersey_y1 = y1 + int(
                    box_height * 0.20
                )

                jersey_y2 = y1 + int(
                    box_height * 0.55
                )

                jersey_x1 = x1 + int(
                    box_width * 0.20
                )

                jersey_x2 = x1 + int(
                    box_width * 0.80
                )


                jersey = frame[
                    jersey_y1:jersey_y2,
                    jersey_x1:jersey_x2
                ]


                if jersey.size == 0:
                    continue


                hsv = cv2.cvtColor(
                    jersey,
                    cv2.COLOR_BGR2HSV
                )

                pixels = hsv.reshape(-1, 3)

                valid = pixels[
                    (pixels[:, 2] > 40) &
                    (pixels[:, 2] < 245)
                ]


                if len(valid) < 20:
                    continue


                player_color = np.mean(
                    valid,
                    axis=0
                )


                # =================================================
                # QUICK TEAM CLUSTERING
                # =================================================

                # We use the dominant jersey hue.
                # Team centers are learned during the first frames.

                if "team_centers" not in globals():

                    team_centers = []


                # =================================================
                # STORE TEMPORARY TEAM SAMPLE
                # =================================================

                if len(team_centers) < 2:

                    team_centers.append(
                        player_color
                    )

                    team = len(team_centers) - 1

                else:

                    distances = np.linalg.norm(
                        np.array(team_centers) -
                        player_color,
                        axis=1
                    )

                    team = int(
                        np.argmin(distances)
                    )


                # =================================================
                # DRAW PLAYER
                # =================================================

                if team == 0:

                    box_color = TEAM_A_COLOR

                else:

                    box_color = TEAM_B_COLOR


                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    box_color,
                    1
                )


                # Activity label
                cv2.putText(
                    frame,
                    activity,
                    (
                        x1,
                        max(15, y1 - 5)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    box_color,
                    1,
                    cv2.LINE_AA
                )


                # =================================================
                # ACTIVITY COUNTS
                # =================================================

                if team == 0:

                    current_team_a[activity] += 1

                else:

                    current_team_b[activity] += 1


    # ========================================================
    # POSSESSION-STYLE ACTIVITY PANEL
    # ========================================================

    panel_x1 = width - 330
    panel_y1 = 20

    panel_x2 = width - 20
    panel_y2 = 230


    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (panel_x1, panel_y1),
        (panel_x2, panel_y2),
        (0, 0, 0),
        -1
    )

    cv2.addWeighted(
        overlay,
        0.60,
        frame,
        0.40,
        0,
        frame
    )


    # ========================================================
    # TITLE
    # ========================================================

    cv2.putText(
        frame,
        "PLAYER ACTIVITY",
        (panel_x1 + 75, panel_y1 + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )


    # ========================================================
    # TEAM A
    # ========================================================

    cv2.putText(
        frame,
        "TEAM A",
        (panel_x1 + 15, panel_y1 + 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        TEAM_A_COLOR,
        1,
        cv2.LINE_AA
    )


    cv2.putText(
        frame,
        f"Standing: {current_team_a['STANDING']}",
        (panel_x1 + 15, panel_y1 + 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        TEAM_A_COLOR,
        1,
        cv2.LINE_AA
    )


    cv2.putText(
        frame,
        f"Walking: {current_team_a['WALKING']}",
        (panel_x1 + 15, panel_y1 + 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        TEAM_A_COLOR,
        1,
        cv2.LINE_AA
    )


    cv2.putText(
        frame,
        f"Jogging: {current_team_a['JOGGING']}",
        (panel_x1 + 15, panel_y1 + 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        TEAM_A_COLOR,
        1,
        cv2.LINE_AA
    )


    cv2.putText(
        frame,
        f"Running: {current_team_a['RUNNING']}",
        (panel_x1 + 15, panel_y1 + 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        TEAM_A_COLOR,
        1,
        cv2.LINE_AA
    )


    # ========================================================
    # TEAM B
    # ========================================================

    cv2.putText(
        frame,
        "TEAM B",
        (panel_x1 + 165, panel_y1 + 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        TEAM_B_COLOR,
        1,
        cv2.LINE_AA
    )


    cv2.putText(
        frame,
        f"Standing: {current_team_b['STANDING']}",
        (panel_x1 + 165, panel_y1 + 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        TEAM_B_COLOR,
        1,
        cv2.LINE_AA
    )


    cv2.putText(
        frame,
        f"Walking: {current_team_b['WALKING']}",
        (panel_x1 + 165, panel_y1 + 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        TEAM_B_COLOR,
        1,
        cv2.LINE_AA
    )


    cv2.putText(
        frame,
        f"Jogging: {current_team_b['JOGGING']}",
        (panel_x1 + 165, panel_y1 + 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        TEAM_B_COLOR,
        1,
        cv2.LINE_AA
    )


    cv2.putText(
        frame,
        f"Running: {current_team_b['RUNNING']}",
        (panel_x1 + 165, panel_y1 + 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        TEAM_B_COLOR,
        1,
        cv2.LINE_AA
    )


    # ========================================================
    # WRITE FRAME
    # ========================================================

    writer.write(frame)


# ============================================================
# FINISH
# ============================================================

cap.release()
writer.release()

print("\n========================================")
print("DONE")
print("========================================")

print("\nOutput saved to:")
print(output_path)

print("========================================")