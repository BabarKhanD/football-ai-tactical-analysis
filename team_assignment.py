from ultralytics import YOLO
import cv2
import numpy as np
from sklearn.cluster import KMeans
import sys
import os


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = r"D:\FootballAI\Traning3\best.pt"

# Display colors in BGR format
TEAM_A_COLOR = (255, 0, 0)         # Blue
TEAM_B_COLOR = (0, 0, 255)         # Red
GOALKEEPER_COLOR = (0, 255, 255)   # Yellow
REFEREE_COLOR = (255, 0, 255)       # Purple
BALL_COLOR = (0, 255, 0)            # Green


# ============================================================
# GET VIDEO FROM COMMAND LINE
# ============================================================

if len(sys.argv) < 2:
    print("Usage:")
    print(
        r'python scripts\team_assignment.py "D:\FootballAI\videos\sequences\SNMOT-140.mp4"'
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
# PASS 1: COLLECT JERSEY COLORS
# ============================================================

print("\nPass 1: Analyzing player jersey colors...")

cap = cv2.VideoCapture(VIDEO_PATH)

colors = []
frame_count = 0

while True:

    success, frame = cap.read()

    if not success:
        break

    frame_count += 1

    # Analyze every 10th frame
    if frame_count % 10 != 0:
        continue

    results = model(
        frame,
        conf=0.25,
        verbose=False
    )

    for box in results[0].boxes:

        cls_id = int(box.cls[0])

        if model.names[cls_id] != "Player":
            continue

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )

        width = x2 - x1
        height = y2 - y1

        if width <= 0 or height <= 0:
            continue

        # ----------------------------------------------------
        # Upper body / jersey region
        # ----------------------------------------------------

        jersey_y1 = y1 + int(height * 0.20)
        jersey_y2 = y1 + int(height * 0.55)

        jersey_x1 = x1 + int(width * 0.20)
        jersey_x2 = x1 + int(width * 0.80)

        jersey = frame[
            jersey_y1:jersey_y2,
            jersey_x1:jersey_x2
        ]

        if jersey.size == 0:
            continue

        # ----------------------------------------------------
        # Convert jersey to HSV
        # ----------------------------------------------------

        hsv = cv2.cvtColor(
            jersey,
            cv2.COLOR_BGR2HSV
        )

        pixels = hsv.reshape(-1, 3)

        # Ignore extremely dark and extremely bright pixels
        valid = pixels[
            (pixels[:, 2] > 40) &
            (pixels[:, 2] < 245)
        ]

        if len(valid) < 20:
            continue

        mean_hsv = np.mean(
            valid,
            axis=0
        )

        colors.append(mean_hsv)


cap.release()


print(f"Frames analyzed: {frame_count}")
print(f"Jersey samples: {len(colors)}")


if len(colors) < 20:
    raise ValueError(
        "Not enough player samples to determine two teams."
    )


# ============================================================
# CLUSTER INTO TWO TEAMS
# ============================================================

print("\nClustering players into two teams...")

kmeans = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
)

kmeans.fit(colors)

team_centers = kmeans.cluster_centers_


print("\nTeam centers (HSV):")
print("Team A:", team_centers[0])
print("Team B:", team_centers[1])


# ============================================================
# PASS 2: CREATE OUTPUT VIDEO
# ============================================================

print("\nPass 2: Creating annotated video...")

cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)

width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)


# ------------------------------------------------------------
# OUTPUT DIRECTORY
# ------------------------------------------------------------

output_dir = r"D:\FootballAI\outputs\team_assignment"

os.makedirs(
    output_dir,
    exist_ok=True
)


video_name = os.path.splitext(
    os.path.basename(VIDEO_PATH)
)[0]


output_path = os.path.join(
    output_dir,
    video_name + "_teams.mp4"
)


# ------------------------------------------------------------
# VIDEO WRITER
# ------------------------------------------------------------

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

writer = cv2.VideoWriter(
    output_path,
    fourcc,
    fps,
    (width, height)
)


# ============================================================
# PROCESS VIDEO
# ============================================================

frame_number = 0

while True:

    success, frame = cap.read()

    if not success:
        break

    frame_number += 1

    results = model(
        frame,
        conf=0.25,
        verbose=False
    )

    for box in results[0].boxes:

        cls_id = int(box.cls[0])

        class_name = model.names[cls_id]

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )


        # ====================================================
        # PLAYER
        # ====================================================

        if class_name == "Player":

            width_box = x2 - x1
            height_box = y2 - y1

            if width_box <= 0 or height_box <= 0:
                continue


            # ------------------------------------------------
            # Jersey region
            # ------------------------------------------------

            jersey_y1 = y1 + int(
                height_box * 0.20
            )

            jersey_y2 = y1 + int(
                height_box * 0.55
            )

            jersey_x1 = x1 + int(
                width_box * 0.20
            )

            jersey_x2 = x1 + int(
                width_box * 0.80
            )


            jersey = frame[
                jersey_y1:jersey_y2,
                jersey_x1:jersey_x2
            ]


            if jersey.size == 0:
                continue


            # ------------------------------------------------
            # HSV jersey color
            # ------------------------------------------------

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


            # ------------------------------------------------
            # Compare with team centers
            # ------------------------------------------------

            distances = np.linalg.norm(
                team_centers - player_color,
                axis=1
            )

            team = np.argmin(distances)


            # ------------------------------------------------
            # Assign team + visualization color
            # ------------------------------------------------

            if team == 0:

                label = "TEAM A"
                box_color = TEAM_A_COLOR

            else:

                label = "TEAM B"
                box_color = TEAM_B_COLOR


            # ------------------------------------------------
            # Player box
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                box_color,
                1
            )


            # ------------------------------------------------
            # Player label
            # ------------------------------------------------

            cv2.putText(
                frame,
                label,
                (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                box_color,
                1,
                cv2.LINE_AA
            )


        # ====================================================
        # GOALKEEPER
        # ====================================================

        elif class_name == "Goalkeeper":

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                GOALKEEPER_COLOR,
                1
            )

            cv2.putText(
                frame,
                "GOALKEEPER",
                (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                GOALKEEPER_COLOR,
                1,
                cv2.LINE_AA
            )


        # ====================================================
        # REFEREE
        # ====================================================

        elif class_name == "Referee":

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                REFEREE_COLOR,
                1
            )

            cv2.putText(
                frame,
                "REFEREE",
                (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                REFEREE_COLOR,
                1,
                cv2.LINE_AA
            )


        # ====================================================
        # BALL
        # ====================================================

        elif class_name == "Ball":

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                BALL_COLOR,
                1
            )

            cv2.putText(
                frame,
                "BALL",
                (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                BALL_COLOR,
                1,
                cv2.LINE_AA
            )


    # --------------------------------------------------------
    # Save frame
    # --------------------------------------------------------

    writer.write(frame)


# ============================================================
# FINISH
# ============================================================

cap.release()
writer.release()

print("\n========================================")
print("DONE")
print("========================================")
print("Output saved to:")
print(output_path)
print("========================================")