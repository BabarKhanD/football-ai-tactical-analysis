```python
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

TEAM_A_COLOR = (255, 0, 0)       # Blue
TEAM_B_COLOR = (0, 0, 255)       # Red
GOALKEEPER_COLOR = (0, 255, 255) # Yellow
REFEREE_COLOR = (255, 0, 255)    # Purple
BALL_COLOR = (0, 255, 0)         # Green

CONFIDENCE = 0.25

# Maximum distance between ball and player's feet
POSSESSION_DISTANCE = 120

# Frames required to confirm a team possession change
CONFIRM_FRAMES = 5

# Frames required to confirm a new internal ball carrier.
# This is ONLY used to count player-to-player changes.
# No Ball Carrier label will be shown.
CARRIER_CONFIRM_FRAMES = 3


# ============================================================
# GET VIDEO PATH
# ============================================================

if len(sys.argv) < 2:

    print("Usage:")
    print(
        r'python scripts\possession.py "D:\FootballAI\videos\sequences\SNMOT-140.mp4"'
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
# PASS 1: LEARN TEAM COLORS
# ============================================================

print("\nPass 1: Learning team jersey colors...")

cap = cv2.VideoCapture(VIDEO_PATH)

colors = []
frame_count = 0


while True:

    success, frame = cap.read()

    if not success:
        break

    frame_count += 1


    # Sample every 10th frame
    if frame_count % 10 != 0:
        continue


    results = model(
        frame,
        conf=CONFIDENCE,
        verbose=False
    )


    for box in results[0].boxes:

        cls_id = int(box.cls[0])

        class_name = model.names[cls_id]


        if class_name != "Player":
            continue


        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )


        box_width = x2 - x1
        box_height = y2 - y1


        if box_width <= 0 or box_height <= 0:
            continue


        # ----------------------------------------------------
        # JERSEY REGION
        # ----------------------------------------------------

        jersey_y1 = y1 + int(box_height * 0.20)
        jersey_y2 = y1 + int(box_height * 0.55)

        jersey_x1 = x1 + int(box_width * 0.20)
        jersey_x2 = x1 + int(box_width * 0.80)


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


        mean_hsv = np.mean(
            valid,
            axis=0
        )


        colors.append(mean_hsv)


cap.release()


print(f"Frames analyzed: {frame_count}")
print(f"Jersey samples collected: {len(colors)}")


if len(colors) < 20:
    raise ValueError("Not enough jersey samples.")


# ============================================================
# CLUSTER INTO TWO TEAMS
# ============================================================

print("\nClustering Team A and Team B...")


kmeans = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
)


kmeans.fit(colors)


team_centers = kmeans.cluster_centers_


print("\nTeam centers:")
print("Team A:", team_centers[0])
print("Team B:", team_centers[1])


# ============================================================
# PASS 2: PROCESS VIDEO
# ============================================================

print("\nPass 2: Calculating possession...")


cap = cv2.VideoCapture(VIDEO_PATH)


fps = cap.get(cv2.CAP_PROP_FPS)


if fps <= 0:
    fps = 25.0


width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)


height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)


# ============================================================
# OUTPUT
# ============================================================

output_dir = r"D:\FootballAI\outputs\possession"


os.makedirs(
    output_dir,
    exist_ok=True
)


video_name = os.path.splitext(
    os.path.basename(VIDEO_PATH)
)[0]


output_path = os.path.join(
    output_dir,
    video_name + "_possession.mp4"
)


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
# POSSESSION VARIABLES
# ============================================================

# Total possession frames
team_a_frames = 0
team_b_frames = 0


# Current confirmed possession team
current_team = None


# Candidate team waiting for confirmation
candidate_team = None
candidate_frames = 0


# Internal player-to-player changes
team_a_changes = 0
team_b_changes = 0


# Current possession streak
current_streak_frames = 0


# ------------------------------------------------------------
# INTERNAL BALL CARRIER TRACKING
# Used ONLY for calculating Changes.
# Nothing related to this will be displayed on the video.
# ------------------------------------------------------------

current_carrier_id = None
current_carrier_team = None

candidate_carrier_id = None
candidate_carrier_team = None
candidate_carrier_frames = 0


# ============================================================
# HELPER FUNCTION: DETERMINE PLAYER TEAM
# ============================================================

def get_team_from_player(frame, x1, y1, x2, y2):

    box_width = x2 - x1
    box_height = y2 - y1


    if box_width <= 0 or box_height <= 0:
        return None


    # Jersey region
    jersey_y1 = y1 + int(box_height * 0.20)
    jersey_y2 = y1 + int(box_height * 0.55)

    jersey_x1 = x1 + int(box_width * 0.20)
    jersey_x2 = x1 + int(box_width * 0.80)


    jersey = frame[
        jersey_y1:jersey_y2,
        jersey_x1:jersey_x2
    ]


    if jersey.size == 0:
        return None


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
        return None


    player_color = np.mean(
        valid,
        axis=0
    )


    distances = np.linalg.norm(
        team_centers - player_color,
        axis=1
    )


    return int(
        np.argmin(distances)
    )


# ============================================================
# PROCESS EACH FRAME
# ============================================================

while True:

    success, frame = cap.read()


    if not success:
        break


    # --------------------------------------------------------
    # TRACK OBJECTS WITH BYTETRACK
    # --------------------------------------------------------

    results = model.track(
        frame,
        conf=CONFIDENCE,
        tracker="bytetrack.yaml",
        persist=True,
        verbose=False
    )


    result = results[0]


    players = []
    ball_center = None


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


            # ------------------------------------------------
            # TRACK ID
            # ------------------------------------------------

            track_id = None


            if box.id is not None:

                track_id = int(
                    box.id[0]
                )


            # =================================================
            # PLAYER
            # =================================================

            if class_name == "Player":

                team = get_team_from_player(
                    frame,
                    x1,
                    y1,
                    x2,
                    y2
                )


                if team is None:
                    continue


                foot_x = (x1 + x2) // 2
                foot_y = y2


                players.append({

                    "id": track_id,
                    "team": team,
                    "foot": (foot_x, foot_y),
                    "box": (x1, y1, x2, y2)

                })


                # --------------------------------------------
                # TEAM COLOR AND LABEL
                # --------------------------------------------

                if team == 0:

                    label = "TEAM A"
                    box_color = TEAM_A_COLOR

                else:

                    label = "TEAM B"
                    box_color = TEAM_B_COLOR


                # Add tracking ID if available
                if track_id is not None:

                    label = f"{label} ID {track_id}"


                # --------------------------------------------
                # DRAW PLAYER BOX
                # --------------------------------------------

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    box_color,
                    1
                )


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


            # =================================================
            # GOALKEEPER
            # =================================================

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


            # =================================================
            # REFEREE
            # =================================================

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


            # =================================================
            # BALL
            # =================================================

            elif class_name == "Ball":

                ball_x = (x1 + x2) // 2
                ball_y = (y1 + y2) // 2


                ball_center = (
                    ball_x,
                    ball_y
                )


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


    # ========================================================
    # FIND CLOSEST PLAYER TO BALL
    # ========================================================

    detected_team = None

    detected_carrier_id = None

    detected_carrier = None


    if ball_center is not None and len(players) > 0:

        closest_distance = float("inf")


        for player in players:

            foot_x, foot_y = player["foot"]


            distance = np.sqrt(

                (ball_center[0] - foot_x) ** 2 +

                (ball_center[1] - foot_y) ** 2

            )


            if distance < closest_distance:

                closest_distance = distance

                detected_carrier = player


        # Ball must be close enough to a player
        if (

            detected_carrier is not None and

            closest_distance <= POSSESSION_DISTANCE

        ):

            detected_team = detected_carrier["team"]

            detected_carrier_id = detected_carrier["id"]


    # ========================================================
    # STABILIZE TEAM POSSESSION
    # ========================================================

    if detected_team is not None:

        if detected_team == current_team:

            candidate_team = None
            candidate_frames = 0


        else:

            if detected_team == candidate_team:

                candidate_frames += 1


            else:

                candidate_team = detected_team

                candidate_frames = 1


            if candidate_frames >= CONFIRM_FRAMES:

                current_team = candidate_team

                candidate_team = None

                candidate_frames = 0


                # Reset current streak
                current_streak_frames = 0


    # ========================================================
    # INTERNAL BALL CARRIER TRACKING
    # USED ONLY TO COUNT TEAM CHANGES
    # NOTHING IS DISPLAYED
    # ========================================================

    if (

        detected_carrier_id is not None and

        detected_team is not None

    ):

        # Same carrier
        if (

            detected_carrier_id == current_carrier_id and

            detected_team == current_carrier_team

        ):

            candidate_carrier_id = None

            candidate_carrier_team = None

            candidate_carrier_frames = 0


        else:

            # New candidate carrier
            if (

                detected_carrier_id == candidate_carrier_id and

                detected_team == candidate_carrier_team

            ):

                candidate_carrier_frames += 1


            else:

                candidate_carrier_id = detected_carrier_id

                candidate_carrier_team = detected_team

                candidate_carrier_frames = 1


            # Confirm carrier change
            if candidate_carrier_frames >= CARRIER_CONFIRM_FRAMES:

                previous_carrier_id = current_carrier_id

                previous_carrier_team = current_carrier_team


                current_carrier_id = candidate_carrier_id

                current_carrier_team = candidate_carrier_team


                candidate_carrier_id = None

                candidate_carrier_team = None

                candidate_carrier_frames = 0


                # --------------------------------------------
                # COUNT INTERNAL TEAM CHANGE
                # --------------------------------------------

                if (

                    previous_carrier_id is not None and

                    previous_carrier_team is not None and

                    previous_carrier_team == current_carrier_team and

                    previous_carrier_id != current_carrier_id

                ):

                    if current_carrier_team == 0:

                        team_a_changes += 1


                    elif current_carrier_team == 1:

                        team_b_changes += 1


    # ========================================================
    # CLEAR CARRIER CANDIDATE WHEN BALL IS NOT RELIABLE
    # ========================================================

    if detected_team is None:

        candidate_carrier_id = None

        candidate_carrier_team = None

        candidate_carrier_frames = 0


    # ========================================================
    # ACCUMULATE POSSESSION TIME
    # ========================================================

    if current_team == 0:

        team_a_frames += 1

        current_streak_frames += 1


    elif current_team == 1:

        team_b_frames += 1

        current_streak_frames += 1


    # ========================================================
    # CALCULATE POSSESSION PERCENTAGES
    # ========================================================

    total_possession_frames = (

        team_a_frames +

        team_b_frames

    )


    if total_possession_frames > 0:

        team_a_percent = (

            team_a_frames /

            total_possession_frames

        ) * 100


        team_b_percent = (

            team_b_frames /

            total_possession_frames

        ) * 100


    else:

        team_a_percent = 0.0

        team_b_percent = 0.0


    # ========================================================
    # CALCULATE TIMES
    # ========================================================

    team_a_seconds = team_a_frames / fps

    team_b_seconds = team_b_frames / fps


    current_streak_seconds = (

        current_streak_frames / fps

    )


    # ========================================================
    # POSSESSION PANEL
    # ========================================================

    panel_x1 = width - 300

    panel_y1 = 20

    panel_x2 = width - 20

    panel_y2 = 215


    # --------------------------------------------------------
    # SEMI-TRANSPARENT BACKGROUND
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    cv2.putText(
        frame,
        "POSSESSION",
        (panel_x1 + 85, panel_y1 + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
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
        (panel_x1 + 15, panel_y1 + 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        TEAM_A_COLOR,
        1,
        cv2.LINE_AA
    )


    cv2.putText(
        frame,
        f"{team_a_percent:.1f}%",
        (panel_x1 + 15, panel_y1 + 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        TEAM_A_COLOR,
        2,
        cv2.LINE_AA
    )


    cv2.putText(
        frame,
        f"Time: {team_a_seconds:.1f}s",
        (panel_x1 + 15, panel_y1 + 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        TEAM_A_COLOR,
        1,
        cv2.LINE_AA
    )


    cv2.putText(
        frame,
        f"Changes: {team_a_changes}",
        (panel_x1 + 15, panel_y1 + 132),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
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
        (panel_x1 + 160, panel_y1 + 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        TEAM_B_COLOR,
        1,
        cv2.LINE_AA
    )


    cv2.putText(
        frame,
        f"{team_b_percent:.1f}%",
        (panel_x1 + 160, panel_y1 + 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        TEAM_B_COLOR,
        2,
        cv2.LINE_AA
    )


    cv2.putText(
        frame,
        f"Time: {team_b_seconds:.1f}s",
        (panel_x1 + 160, panel_y1 + 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        TEAM_B_COLOR,
        1,
        cv2.LINE_AA
    )


    cv2.putText(
        frame,
        f"Changes: {team_b_changes}",
        (panel_x1 + 160, panel_y1 + 132),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        TEAM_B_COLOR,
        1,
        cv2.LINE_AA
    )


    # ========================================================
    # CURRENT POSSESSION
    # ========================================================

    if current_team == 0:

        possession_text = "TEAM A IN POSSESSION"

        possession_color = TEAM_A_COLOR


    elif current_team == 1:

        possession_text = "TEAM B IN POSSESSION"

        possession_color = TEAM_B_COLOR


    else:

        possession_text = "NO POSSESSION"

        possession_color = (255, 255, 255)


    cv2.putText(
        frame,
        possession_text,
        (panel_x1 + 15, panel_y1 + 165),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        possession_color,
        1,
        cv2.LINE_AA
    )


    # ========================================================
    # CURRENT STREAK
    # ========================================================

    cv2.putText(
        frame,
        f"Current streak: {current_streak_seconds:.1f}s",
        (panel_x1 + 15, panel_y1 + 190),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )


    # ========================================================
    # SAVE FRAME
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


print(
    f"Team A possession time: "
    f"{team_a_seconds:.2f} seconds"
)


print(
    f"Team B possession time: "
    f"{team_b_seconds:.2f} seconds"
)


print(
    f"Team A possession: "
    f"{team_a_percent:.2f}%"
)


print(
    f"Team B possession: "
    f"{team_b_percent:.2f}%"
)


print(
    f"Team A player changes: "
    f"{team_a_changes}"
)


print(
    f"Team B player changes: "
    f"{team_b_changes}"
)


print("\nOutput saved to:")

print(output_path)


print("========================================")
```
