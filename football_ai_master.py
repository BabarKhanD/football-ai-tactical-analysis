from ultralytics import YOLO
import cv2
import numpy as np
from sklearn.cluster import KMeans
import os
import sys
import glob
import re
import math


# ============================================================
# FOOTBALL AI MASTER PIPELINE
# ============================================================

MODEL_PATH = r"D:\FootballAI\Traning3\best.pt"

SOCCERNET_ROOT = r"D:\FootballAI\soccernet\tracking\test"

SEQUENCE_VIDEO_DIR = r"D:\FootballAI\videos\sequences"

OUTPUT_DIR = r"D:\FootballAI\outputs\master"


# ============================================================
# COLORS - OpenCV uses BGR
# ============================================================

TEAM_A_COLOR = (255, 0, 0)        # Blue
TEAM_B_COLOR = (0, 0, 255)        # Red

GOALKEEPER_COLOR = (0, 255, 255)  # Yellow
REFEREE_COLOR = (255, 0, 255)     # Purple
BALL_COLOR = (0, 255, 0)         # Green

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


# ============================================================
# YOLO
# ============================================================

CONFIDENCE = 0.25


# ============================================================
# POSSESSION
# ============================================================

POSSESSION_DISTANCE = 120

# Number of frames required to confirm a team change
CONFIRM_FRAMES = 5


# ============================================================
# ACTIVITY
# ============================================================

# Movement thresholds in pixels/frame

STANDING_THRESHOLD = 1.0
WALKING_THRESHOLD = 3.0
JOGGING_THRESHOLD = 7.0

# Maximum distance for matching a detection with
# a nearby detection from the previous frame.
#
# IMPORTANT:
# This is NOT persistent player ID tracking.
# It is only used to estimate current movement.

ACTIVITY_MATCH_DISTANCE = 100


# ============================================================
# HELPER: NATURAL SORT
# ============================================================

def natural_sort_key(path):

    filename = os.path.basename(path)

    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", filename)
    ]


# ============================================================
# STEP 1
# FIND FRAMES
# ============================================================

def find_sequence_frames(sequence_number):

    sequence_name = f"SNMOT-{sequence_number}"

    frame_dir = os.path.join(
        SOCCERNET_ROOT,
        sequence_name,
        "img1"
    )

    print("\nSearching for SoccerNet frames...")
    print(frame_dir)

    if not os.path.exists(frame_dir):

        print("\nERROR: Frame folder not found.")

        print(
            f"\nExpected:\n"
            f"{frame_dir}"
        )

        return None

    frame_files = []

    for extension in ["*.jpg", "*.jpeg", "*.png"]:

        frame_files.extend(
            glob.glob(
                os.path.join(
                    frame_dir,
                    extension
                )
            )
        )

    frame_files = sorted(
        frame_files,
        key=natural_sort_key
    )

    if len(frame_files) == 0:

        print("\nERROR: No images found.")

        return None

    print(
        f"Found {len(frame_files)} frames."
    )

    return frame_files


# ============================================================
# STEP 2
# CREATE VIDEO FROM FRAMES
# ============================================================

def create_video_from_frames(
    frame_files,
    sequence_number
):

    sequence_name = f"SNMOT-{sequence_number}"

    os.makedirs(
        SEQUENCE_VIDEO_DIR,
        exist_ok=True
    )

    video_path = os.path.join(
        SEQUENCE_VIDEO_DIR,
        sequence_name + ".mp4"
    )

    print("\n==============================================")
    print("CREATING VIDEO FROM SOCCERNET FRAMES")
    print("==============================================")

    first_frame = cv2.imread(
        frame_files[0]
    )

    if first_frame is None:

        raise ValueError(
            "Could not read first frame."
        )

    height, width = first_frame.shape[:2]

    fps = 25.0

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        video_path,
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():

        raise RuntimeError(
            "Could not create video writer."
        )

    for index, frame_path in enumerate(
        frame_files,
        start=1
    ):

        frame = cv2.imread(
            frame_path
        )

        if frame is None:
            continue

        # Make sure all frames have
        # exactly the same resolution.

        if (
            frame.shape[1] != width
            or frame.shape[0] != height
        ):

            frame = cv2.resize(
                frame,
                (width, height)
            )

        writer.write(frame)

        if index % 100 == 0:

            print(
                f"Video frames created: "
                f"{index}/{len(frame_files)}"
            )

    writer.release()

    print("\nVideo created successfully.")

    print(
        f"Saved to:\n{video_path}"
    )

    return video_path


# ============================================================
# STEP 3
# TEAM COLOR LEARNING
# ============================================================

def learn_team_colors(video_path, model):

    print("\n==============================================")
    print("PASS 1 - LEARNING TEAM COLORS")
    print("==============================================")

    cap = cv2.VideoCapture(
        video_path
    )

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

        if len(results) == 0:
            continue

        result = results[0]

        if result.boxes is None:
            continue

        for box in result.boxes:

            cls_id = int(
                box.cls[0]
            )

            class_name = model.names[
                cls_id
            ]

            if class_name != "Player":
                continue

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            box_width = x2 - x1
            box_height = y2 - y1

            if (
                box_width <= 0
                or box_height <= 0
            ):
                continue

            # Jersey region

            jersey_y1 = (
                y1 +
                int(box_height * 0.20)
            )

            jersey_y2 = (
                y1 +
                int(box_height * 0.55)
            )

            jersey_x1 = (
                x1 +
                int(box_width * 0.20)
            )

            jersey_x2 = (
                x1 +
                int(box_width * 0.80)
            )

            # Clamp coordinates

            jersey_x1 = max(
                0,
                jersey_x1
            )

            jersey_y1 = max(
                0,
                jersey_y1
            )

            jersey_x2 = min(
                frame.shape[1],
                jersey_x2
            )

            jersey_y2 = min(
                frame.shape[0],
                jersey_y2
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

            pixels = hsv.reshape(
                -1,
                3
            )

            valid = pixels[
                (pixels[:, 2] > 40)
                &
                (pixels[:, 2] < 245)
            ]

            if len(valid) < 20:
                continue

            mean_hsv = np.mean(
                valid,
                axis=0
            )

            colors.append(
                mean_hsv
            )

    cap.release()

    print(
        f"Frames analyzed: {frame_count}"
    )

    print(
        f"Jersey samples collected: "
        f"{len(colors)}"
    )

    if len(colors) < 20:

        raise ValueError(
            "Not enough jersey samples "
            "to determine the two teams."
        )

    # --------------------------------------------------------
    # K-MEANS
    # --------------------------------------------------------

    kmeans = KMeans(
        n_clusters=2,
        random_state=42,
        n_init=10
    )

    kmeans.fit(colors)

    team_centers = (
        kmeans.cluster_centers_
    )

    print("\nTeam centers (HSV):")

    print(
        "Team A:",
        team_centers[0]
    )

    print(
        "Team B:",
        team_centers[1]
    )

    return team_centers


# ============================================================
# HELPER
# GET TEAM FROM PLAYER JERSEY
# ============================================================

def get_team_from_player(
    frame,
    x1,
    y1,
    x2,
    y2,
    team_centers
):

    box_width = x2 - x1
    box_height = y2 - y1

    if (
        box_width <= 0
        or box_height <= 0
    ):
        return None

    jersey_y1 = (
        y1 +
        int(box_height * 0.20)
    )

    jersey_y2 = (
        y1 +
        int(box_height * 0.55)
    )

    jersey_x1 = (
        x1 +
        int(box_width * 0.20)
    )

    jersey_x2 = (
        x1 +
        int(box_width * 0.80)
    )

    jersey_x1 = max(
        0,
        jersey_x1
    )

    jersey_y1 = max(
        0,
        jersey_y1
    )

    jersey_x2 = min(
        frame.shape[1],
        jersey_x2
    )

    jersey_y2 = min(
        frame.shape[0],
        jersey_y2
    )

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

    pixels = hsv.reshape(
        -1,
        3
    )

    valid = pixels[
        (pixels[:, 2] > 40)
        &
        (pixels[:, 2] < 245)
    ]

    if len(valid) < 20:
        return None

    player_color = np.mean(
        valid,
        axis=0
    )

    distances = np.linalg.norm(
        team_centers -
        player_color,
        axis=1
    )

    return int(
        np.argmin(distances)
    )


# ============================================================
# ACTIVITY CLASSIFICATION
# ============================================================

def classify_activity(speed):

    if speed < STANDING_THRESHOLD:

        return "Standing"

    elif speed < WALKING_THRESHOLD:

        return "Walking"

    elif speed < JOGGING_THRESHOLD:

        return "Jogging"

    else:

        return "Running"


# ============================================================
# ACTIVITY MATCHING
# ============================================================

def calculate_activity(
    current_players,
    previous_players
):

    # Separate activity counts
    # for Team A and Team B.

    activity_counts = {

        0: {
            "Standing": 0,
            "Walking": 0,
            "Jogging": 0,
            "Running": 0
        },

        1: {
            "Standing": 0,
            "Walking": 0,
            "Jogging": 0,
            "Running": 0
        }
    }

    # Activity for each current player.
    #
    # The list position corresponds to
    # the current players list.
    #
    # No persistent player ID is used.

    player_activities = []

    used_previous = set()

    for player in current_players:

        cx = player["center"][0]
        cy = player["center"][1]

        team = player["team"]

        best_index = None
        best_distance = float("inf")

        for index, previous in enumerate(
            previous_players
        ):

            if index in used_previous:
                continue

            # Only compare players from
            # the same team.

            if previous["team"] != team:
                continue

            px = previous["center"][0]
            py = previous["center"][1]

            distance = math.sqrt(
                (cx - px) ** 2
                +
                (cy - py) ** 2
            )

            if distance < best_distance:

                best_distance = distance
                best_index = index

        if (
            best_index is not None
            and
            best_distance <=
            ACTIVITY_MATCH_DISTANCE
        ):

            speed = best_distance

            used_previous.add(
                best_index
            )

        else:

            # New/unmatched player.
            #
            # We do not make a strong
            # movement assumption.

            speed = 0

        activity = classify_activity(
            speed
        )

        player_activities.append(
            activity
        )

        activity_counts[
            team
        ][
            activity
        ] += 1

    return (
        player_activities,
        activity_counts
    )


# ============================================================
# DRAW TEXT
# ============================================================

def draw_text(
    frame,
    text,
    position,
    color,
    scale=0.4,
    thickness=1
):

    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )


# ============================================================
# STEP 4
# COMPLETE VIDEO ANALYSIS
# ============================================================

def process_video(
    video_path,
    team_centers,
    model,
    sequence_number
):

    print("\n==============================================")
    print("PASS 2 - COMPLETE FOOTBALL AI ANALYSIS")
    print("==============================================")

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    cap = cv2.VideoCapture(
        video_path
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 25.0

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        f"SNMOT-{sequence_number}_FINAL.mp4"
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

    # ========================================================
    # POSSESSION VARIABLES
    # ========================================================

    team_a_frames = 0
    team_b_frames = 0

    current_team = None

    candidate_team = None
    candidate_frames = 0

    current_streak_frames = 0

    # --------------------------------------------------------
    # INTERNAL TEAM CHANGES
    #
    # We use the nearest current player to the ball.
    #
    # This is NOT used as a permanent player identity.
    # --------------------------------------------------------

    previous_possession_player = None

    team_a_changes = 0
    team_b_changes = 0

    # ========================================================
    # ACTIVITY
    # ========================================================

    previous_players = []

    # ========================================================
    # FRAME LOOP
    # ========================================================

    frame_number = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame_number += 1

        results = model.track(
            frame,
            conf=CONFIDENCE,
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False
        )

        if len(results) == 0:

            writer.write(frame)

            continue

        result = results[0]

        players = []

        ball_center = None

        # ====================================================
        # DETECTIONS
        # ====================================================

        if result.boxes is not None:

            for box in result.boxes:

                cls_id = int(
                    box.cls[0]
                )

                class_name = (
                    model.names[
                        cls_id
                    ]
                )

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                # =================================================
                # PLAYER
                # =================================================

                if class_name == "Player":

                    team = (
                        get_team_from_player(
                            frame,
                            x1,
                            y1,
                            x2,
                            y2,
                            team_centers
                        )
                    )

                    if team is None:
                        continue

                    center_x = (
                        x1 + x2
                    ) // 2

                    center_y = (
                        y1 + y2
                    ) // 2

                    foot_x = center_x
                    foot_y = y2

                    players.append(
                        {
                            "team": team,

                            "center": (
                                center_x,
                                center_y
                            ),

                            "foot": (
                                foot_x,
                                foot_y
                            ),

                            "box": (
                                x1,
                                y1,
                                x2,
                                y2
                            )
                        }
                    )

                    if team == 0:

                        label = "TEAM A"

                        box_color = (
                            TEAM_A_COLOR
                        )

                    else:

                        label = "TEAM B"

                        box_color = (
                            TEAM_B_COLOR
                        )

                    # ------------------------------------------------
                    # PLAYER BOX
                    # ------------------------------------------------

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        box_color,
                        1
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

                    draw_text(
                        frame,
                        "GOALKEEPER",
                        (
                            x1,
                            max(
                                15,
                                y1 - 5
                            )
                        ),
                        GOALKEEPER_COLOR,
                        0.4,
                        1
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

                    draw_text(
                        frame,
                        "REFEREE",
                        (
                            x1,
                            max(
                                15,
                                y1 - 5
                            )
                        ),
                        REFEREE_COLOR,
                        0.4,
                        1
                    )

                # =================================================
                # BALL
                # =================================================

                elif class_name == "Ball":

                    ball_x = (
                        x1 + x2
                    ) // 2

                    ball_y = (
                        y1 + y2
                    ) // 2

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

                    draw_text(
                        frame,
                        "BALL",
                        (
                            x1,
                            max(
                                15,
                                y1 - 5
                            )
                        ),
                        BALL_COLOR,
                        0.4,
                        1
                    )

        # ====================================================
        # ACTIVITY
        # ====================================================

        (
            player_activities,
            activity_counts
        ) = calculate_activity(
            players,
            previous_players
        )

        # ====================================================
        # DRAW PLAYER ACTIVITY + TEAM LABEL
        #
        # Activity is displayed above TEAM A / TEAM B.
        #
        # No player ID is displayed.
        # ====================================================

        for index, player in enumerate(
            players
        ):

            x1, y1, x2, y2 = (
                player["box"]
            )

            team = player["team"]

            if team == 0:

                box_color = (
                    TEAM_A_COLOR
                )

                team_label = "TEAM A"

            else:

                box_color = (
                    TEAM_B_COLOR
                )

                team_label = "TEAM B"

            activity = (
                player_activities[index]
            )

            # ------------------------------------------------
            # ACTIVITY
            # ------------------------------------------------

            draw_text(
                frame,
                activity,
                (
                    x1,
                    max(
                        15,
                        y1 - 22
                    )
                ),
                WHITE,
                0.35,
                1
            )

            # ------------------------------------------------
            # TEAM
            # ------------------------------------------------

            draw_text(
                frame,
                team_label,
                (
                    x1,
                    max(
                        30,
                        y1 - 5
                    )
                ),
                box_color,
                0.4,
                1
            )

        # ====================================================
        # SAVE CURRENT PLAYERS FOR NEXT FRAME
        # ====================================================

        previous_players = [
            {
                "center": p["center"],
                "team": p["team"]
            }
            for p in players
        ]

        # ====================================================
        # FIND PLAYER CLOSEST TO BALL
        # ====================================================

        detected_team = None

        closest_player_index = None

        if (
            ball_center is not None
            and
            len(players) > 0
        ):

            closest_distance = float(
                "inf"
            )

            for index, player in enumerate(
                players
            ):

                foot_x, foot_y = (
                    player["foot"]
                )

                distance = math.sqrt(
                    (
                        ball_center[0]
                        - foot_x
                    ) ** 2
                    +
                    (
                        ball_center[1]
                        - foot_y
                    ) ** 2
                )

                if distance < closest_distance:

                    closest_distance = (
                        distance
                    )

                    closest_player_index = (
                        index
                    )

            if (
                closest_player_index
                is not None
                and
                closest_distance
                <= POSSESSION_DISTANCE
            ):

                detected_team = (
                    players[
                        closest_player_index
                    ]["team"]
                )

        # ====================================================
        # STABILIZE TEAM POSSESSION
        # ====================================================

        if detected_team is not None:

            if detected_team == current_team:

                candidate_team = None
                candidate_frames = 0

            else:

                if (
                    detected_team
                    ==
                    candidate_team
                ):

                    candidate_frames += 1

                else:

                    candidate_team = (
                        detected_team
                    )

                    candidate_frames = 1

                if (
                    candidate_frames
                    >= CONFIRM_FRAMES
                ):

                    # ----------------------------------------
                    # TEAM CHANGED
                    # ----------------------------------------

                    current_team = (
                        candidate_team
                    )

                    candidate_team = None
                    candidate_frames = 0

                    current_streak_frames = 0

                    previous_possession_player = (
                        None
                    )

        # ====================================================
        # INTERNAL SAME-TEAM CHANGES
        # ====================================================

        if (
            detected_team is not None
            and
            closest_player_index is not None
        ):

            current_player_position = (
                players[
                    closest_player_index
                ]["center"]
            )

            if (
                current_team
                ==
                detected_team
            ):

                if (
                    previous_possession_player
                    is not None
                ):

                    previous_x = (
                        previous_possession_player[0]
                    )

                    previous_y = (
                        previous_possession_player[1]
                    )

                    current_x = (
                        current_player_position[0]
                    )

                    current_y = (
                        current_player_position[1]
                    )

                    player_switch_distance = (
                        math.sqrt(
                            (
                                current_x
                                - previous_x
                            ) ** 2
                            +
                            (
                                current_y
                                - previous_y
                            ) ** 2
                        )
                    )

                    # A meaningful change in
                    # closest player location
                    # indicates a likely
                    # same-team pass/change.

                    if (
                        player_switch_distance
                        > 35
                    ):

                        if current_team == 0:

                            team_a_changes += 1

                        elif current_team == 1:

                            team_b_changes += 1

                previous_possession_player = (
                    current_player_position
                )

        # ====================================================
        # POSSESSION TIME
        # ====================================================

        if current_team == 0:

            team_a_frames += 1

            current_streak_frames += 1

        elif current_team == 1:

            team_b_frames += 1

            current_streak_frames += 1

        # ====================================================
        # POSSESSION PERCENTAGE
        # ====================================================

        total_possession_frames = (
            team_a_frames
            +
            team_b_frames
        )

        if (
            total_possession_frames
            > 0
        ):

            team_a_percent = (
                team_a_frames
                /
                total_possession_frames
            ) * 100

            team_b_percent = (
                team_b_frames
                /
                total_possession_frames
            ) * 100

        else:

            team_a_percent = 0.0
            team_b_percent = 0.0

        # ====================================================
        # TIME
        # ====================================================

        team_a_seconds = (
            team_a_frames
            /
            fps
        )

        team_b_seconds = (
            team_b_frames
            /
            fps
        )

        current_streak_seconds = (
            current_streak_frames
            /
            fps
        )

        # ====================================================
        # POSSESSION PANEL
        # ====================================================

        panel_x1 = width - 340
        panel_y1 = 20

        panel_x2 = width - 20
        panel_y2 = 390

        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (
                panel_x1,
                panel_y1
            ),
            (
                panel_x2,
                panel_y2
            ),
            BLACK,
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

        # ====================================================
        # TITLE
        # ====================================================

        draw_text(
            frame,
            "FOOTBALL AI",
            (
                panel_x1 + 105,
                panel_y1 + 25
            ),
            WHITE,
            0.6,
            1
        )

        draw_text(
            frame,
            "POSSESSION",
            (
                panel_x1 + 110,
                panel_y1 + 50
            ),
            WHITE,
            0.5,
            1
        )

        # ====================================================
        # TEAM A
        # ====================================================

        draw_text(
            frame,
            "TEAM A",
            (
                panel_x1 + 15,
                panel_y1 + 85
            ),
            TEAM_A_COLOR,
            0.5,
            1
        )

        draw_text(
            frame,
            f"{team_a_percent:.1f}%",
            (
                panel_x1 + 15,
                panel_y1 + 112
            ),
            TEAM_A_COLOR,
            0.65,
            2
        )

        draw_text(
            frame,
            f"Time: {team_a_seconds:.1f}s",
            (
                panel_x1 + 15,
                panel_y1 + 138
            ),
            TEAM_A_COLOR,
            0.4,
            1
        )

        draw_text(
            frame,
            f"Changes: {team_a_changes}",
            (
                panel_x1 + 15,
                panel_y1 + 160
            ),
            TEAM_A_COLOR,
            0.4,
            1
        )

        # ====================================================
        # TEAM B
        # ====================================================

        draw_text(
            frame,
            "TEAM B",
            (
                panel_x1 + 175,
                panel_y1 + 85
            ),
            TEAM_B_COLOR,
            0.5,
            1
        )

        draw_text(
            frame,
            f"{team_b_percent:.1f}%",
            (
                panel_x1 + 175,
                panel_y1 + 112
            ),
            TEAM_B_COLOR,
            0.65,
            2
        )

        draw_text(
            frame,
            f"Time: {team_b_seconds:.1f}s",
            (
                panel_x1 + 175,
                panel_y1 + 138
            ),
            TEAM_B_COLOR,
            0.4,
            1
        )

        draw_text(
            frame,
            f"Changes: {team_b_changes}",
            (
                panel_x1 + 175,
                panel_y1 + 160
            ),
            TEAM_B_COLOR,
            0.4,
            1
        )

        # ====================================================
        # CURRENT POSSESSION
        # ====================================================

        if current_team == 0:

            possession_text = (
                "TEAM A IN POSSESSION"
            )

            possession_color = (
                TEAM_A_COLOR
            )

        elif current_team == 1:

            possession_text = (
                "TEAM B IN POSSESSION"
            )

            possession_color = (
                TEAM_B_COLOR
            )

        else:

            possession_text = (
                "NO POSSESSION"
            )

            possession_color = WHITE

        draw_text(
            frame,
            possession_text,
            (
                panel_x1 + 15,
                panel_y1 + 195
            ),
            possession_color,
            0.45,
            1
        )

        # ====================================================
        # CURRENT STREAK
        # ====================================================

        draw_text(
            frame,
            f"Current streak: "
            f"{current_streak_seconds:.1f}s",
            (
                panel_x1 + 15,
                panel_y1 + 220
            ),
            WHITE,
            0.4,
            1
        )

        # ====================================================
        # ACTIVITY PANEL
        # ====================================================

        draw_text(
            frame,
            "PLAYER ACTIVITY",
            (
                panel_x1 + 85,
                panel_y1 + 250
            ),
            WHITE,
            0.5,
            1
        )

        # ====================================================
        # TEAM A ACTIVITY
        # ====================================================

        draw_text(
            frame,
            "TEAM A",
            (
                panel_x1 + 15,
                panel_y1 + 275
            ),
            TEAM_A_COLOR,
            0.4,
            1
        )

        draw_text(
            frame,
            f"Standing: "
            f"{activity_counts[0]['Standing']}",
            (
                panel_x1 + 15,
                panel_y1 + 295
            ),
            WHITE,
            0.35,
            1
        )

        draw_text(
            frame,
            f"Walking: "
            f"{activity_counts[0]['Walking']}",
            (
                panel_x1 + 15,
                panel_y1 + 315
            ),
            WHITE,
            0.35,
            1
        )

        draw_text(
            frame,
            f"Jogging: "
            f"{activity_counts[0]['Jogging']}",
            (
                panel_x1 + 15,
                panel_y1 + 335
            ),
            WHITE,
            0.35,
            1
        )

        draw_text(
            frame,
            f"Running: "
            f"{activity_counts[0]['Running']}",
            (
                panel_x1 + 15,
                panel_y1 + 355
            ),
            WHITE,
            0.35,
            1
        )

        # ====================================================
        # TEAM B ACTIVITY
        # ====================================================

        draw_text(
            frame,
            "TEAM B",
            (
                panel_x1 + 175,
                panel_y1 + 275
            ),
            TEAM_B_COLOR,
            0.4,
            1
        )

        draw_text(
            frame,
            f"Standing: "
            f"{activity_counts[1]['Standing']}",
            (
                panel_x1 + 175,
                panel_y1 + 295
            ),
            WHITE,
            0.35,
            1
        )

        draw_text(
            frame,
            f"Walking: "
            f"{activity_counts[1]['Walking']}",
            (
                panel_x1 + 175,
                panel_y1 + 315
            ),
            WHITE,
            0.35,
            1
        )

        draw_text(
            frame,
            f"Jogging: "
            f"{activity_counts[1]['Jogging']}",
            (
                panel_x1 + 175,
                panel_y1 + 335
            ),
            WHITE,
            0.35,
            1
        )

        draw_text(
            frame,
            f"Running: "
            f"{activity_counts[1]['Running']}",
            (
                panel_x1 + 175,
                panel_y1 + 355
            ),
            WHITE,
            0.35,
            1
        )

        # ====================================================
        # SAVE
        # ====================================================

        writer.write(frame)

        if frame_number % 100 == 0:

            print(
                f"Processed frames: "
                f"{frame_number}"
            )

    # ========================================================
    # FINISH
    # ========================================================

    cap.release()
    writer.release()

    print("\n==============================================")
    print("MASTER ANALYSIS COMPLETE")
    print("==============================================")

    print(
        f"Team A possession: "
        f"{team_a_percent:.2f}%"
    )

    print(
        f"Team B possession: "
        f"{team_b_percent:.2f}%"
    )

    print(
        f"Team A possession time: "
        f"{team_a_seconds:.2f}s"
    )

    print(
        f"Team B possession time: "
        f"{team_b_seconds:.2f}s"
    )

    print(
        f"Team A changes: "
        f"{team_a_changes}"
    )

    print(
        f"Team B changes: "
        f"{team_b_changes}"
    )

    print("\nFINAL VIDEO:")

    print(output_path)

    return output_path


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")

    print("==============================================")
    print("       FOOTBALL AI MASTER PIPELINE")
    print("==============================================")

    print()

    print("This pipeline will:")
    print()

    print("1. Find SoccerNet frames")
    print("2. Create the sequence video")
    print("3. Learn Team A / Team B")
    print("4. Run YOLO detection")
    print("5. Run ByteTrack")
    print("6. Apply team classification")
    print("7. Draw colored object boxes")
    print("8. Calculate possession")
    print("9. Calculate possession time")
    print("10. Calculate possession changes")
    print("11. Calculate current streak")
    print("12. Calculate player activity")
    print("13. Show activity above each player")
    print("14. Show team activity statistics")
    print("15. Create final annotated video")

    print()

    print("==============================================")

    # ========================================================
    # ASK SEQUENCE
    # ========================================================

    sequence_number = input(
        "\nEnter sequence number "
        "(example: 140): "
    ).strip()

    if not sequence_number.isdigit():

        print(
            "\nERROR: Sequence number must "
            "be numeric."
        )

        sys.exit(1)

    # ========================================================
    # MODEL
    # ========================================================

    print("\nLoading YOLO model...")

    if not os.path.exists(
        MODEL_PATH
    ):

        print(
            "\nERROR: YOLO model not found:"
        )

        print(MODEL_PATH)

        sys.exit(1)

    model = YOLO(
        MODEL_PATH
    )

    print(
        "YOLO model loaded successfully."
    )

    # ========================================================
    # FIND FRAMES
    # ========================================================

    frame_files = find_sequence_frames(
        sequence_number
    )

    if frame_files is None:

        sys.exit(1)

    # ========================================================
    # CREATE VIDEO
    # ========================================================

    video_path = create_video_from_frames(
        frame_files,
        sequence_number
    )

    # ========================================================
    # LEARN TEAMS
    # ========================================================

    team_centers = learn_team_colors(
        video_path,
        model
    )

    # ========================================================
    # COMPLETE ANALYSIS
    # ========================================================

    final_video = process_video(
        video_path,
        team_centers,
        model,
        sequence_number
    )

    # ========================================================
    # FINAL MESSAGE
    # ========================================================

    print("\n")

    print("==============================================")
    print("          EVERYTHING IS COMPLETE")
    print("==============================================")

    print(
        "\nSequence video:"
    )

    print(video_path)

    print(
        "\nFinal Football AI video:"
    )

    print(final_video)

    print(
        "\n=============================================="
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()