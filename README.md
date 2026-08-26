# ⚽ Football AI Tactical Analysis

> **An AI-powered football video analytics system built using YOLO, OpenCV, ByteTrack, and machine learning techniques to detect football entities, classify teams, analyze possession, and estimate real-time player activity.**

---

## 🚀 Project Overview

**Football AI Tactical Analysis** is a computer vision project designed to analyze football match footage and generate useful tactical and gameplay insights automatically.

The system processes football video frame by frame and detects:

* Players
* Goalkeepers
* Referees
* Football

It then performs additional analysis including **team classification, ball possession estimation, possession statistics, possession changes, current possession streaks, and player activity recognition**.

The final result is an annotated football video displaying detections and real-time analytics.

---

# 🎯 Features

## 🔍 Object Detection

The system uses a custom-trained **YOLO model** to detect four football-related classes:

| Class         | Description              |
| ------------- | ------------------------ |
| ⚽ Ball        | Detects the football     |
| 🧍 Player     | Detects football players |
| 🧤 Goalkeeper | Detects goalkeepers      |
| 🟪 Referee    | Detects match referees   |

---

## 👥 Automatic Team Classification

Players are automatically separated into **Team A** and **Team B**.

The system:

1. Extracts the jersey region from detected players.
2. Converts jersey colors into HSV color space.
3. Collects jersey color samples.
4. Uses **K-Means clustering** to identify two dominant team groups.
5. Assigns each detected player to Team A or Team B.

Each team receives a separate bounding-box color.

---

## ⚽ Ball Possession Analysis

The system estimates ball possession by finding the player closest to the detected ball.

Possession information includes:

* Team A possession percentage
* Team B possession percentage
* Possession time for both teams
* Current team in possession
* Current possession streak
* Estimated possession changes

A stabilization mechanism is used to reduce rapid possession switching caused by temporary detection errors.

---

## 🏃 Player Activity Recognition

The system estimates the current activity of detected players without relying on permanent player identities.

Player movement between consecutive frames is used to estimate motion speed and classify activity as:

* 🧍 Standing
* 🚶 Walking
* 🏃 Jogging
* 🏃‍♂️ Running

This information can be displayed alongside the detected player and used for real-time activity analysis.

---

## 🎥 Automatic Video Generation

The master pipeline can automatically:

1. Locate a selected sequence of football frames.
2. Convert the frames into an MP4 video.
3. Learn the two team jersey color groups.
4. Run YOLO detection.
5. Apply object tracking.
6. Classify players into Team A and Team B.
7. Estimate player activity.
8. Estimate ball possession.
9. Generate possession statistics.
10. Produce a final annotated football analysis video.

The user simply enters a sequence number.

```text
Enter sequence number (example: 140): 123
```

---

# 🧠 System Pipeline

```text
Football Frames
       │
       ▼
Video Generation
       │
       ▼
YOLO Object Detection
       │
       ├── Players
       ├── Goalkeepers
       ├── Referees
       └── Ball
       │
       ▼
Team Classification
       │
       ├── Jersey Extraction
       ├── HSV Color Analysis
       └── K-Means Clustering
       │
       ▼
ByteTrack
       │
       ▼
Player Activity Estimation
       │
       ├── Standing
       ├── Walking
       ├── Jogging
       └── Running
       │
       ▼
Ball Proximity Analysis
       │
       ▼
Possession Estimation
       │
       ├── Team A %
       ├── Team B %
       ├── Possession Time
       ├── Current Possession
       ├── Current Streak
       └── Possession Changes
       │
       ▼
Annotated Football Analytics Video
```

---

# 🛠️ Technologies Used

* Python
* YOLO
* Ultralytics
* OpenCV
* NumPy
* Scikit-learn
* ByteTrack
* K-Means Clustering

---

# 📂 Project Structure

```text
football-ai-tactical-analysis/
│
├── scripts/
│   └── football_ai_master.py
│
├── models/
│   └── best.pt
│
├── outputs/
│   └── master/
│
├── videos/
│   └── sequences/
│
├── requirements.txt
│
└── README.md
```

> Dataset files and original football footage are not included in this repository.

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/football-ai-tactical-analysis.git
cd football-ai-tactical-analysis
```

## 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

Run the master pipeline:

```bash
python scripts/football_ai_master.py
```

The program will ask for a sequence number:

```text
Enter sequence number (example: 140):
```

The pipeline will then:

```text
[1] Locate football frames
[2] Generate video
[3] Learn team jersey colors
[4] Detect players, goalkeepers, referees and ball
[5] Apply tracking
[6] Classify Team A and Team B
[7] Estimate player activity
[8] Calculate ball possession
[9] Generate statistics
[10] Create final annotated video
```

The final video is saved inside:

```text
outputs/master/
```

---

# 📊 Analytics Produced

The final output can include:

### Team Classification

```text
TEAM A
TEAM B
```

### Possession

```text
Team A: XX.X%
Team B: XX.X%

Team A Time: XX.X seconds
Team B Time: XX.X seconds

Current Team in Possession
Current Possession Streak

Possession Changes
```

### Player Activity

```text
Standing
Walking
Jogging
Running
```

---

# 📌 Important Note About the Dataset

This repository does **not** contain SoccerNet videos, frames, annotations, or any other restricted dataset material.

Football footage used during development was accessed separately under the applicable SoccerNet access terms and Non-Disclosure Agreement.

If you want to use SoccerNet data, please obtain access directly from the official SoccerNet project:

[SoccerNet Official Website](https://www.soccer-net.org/?utm_source=chatgpt.com)

This repository only contains the project implementation and code.

---

# 🔮 Future Improvements

Possible future improvements include:

* Persistent player re-identification
* Individual player distance covered
* Player-specific movement statistics
* Player trajectory visualization
* Tactical heatmaps
* Passing analysis
* Formation detection
* Player tracking across camera cuts
* Advanced ball trajectory analysis
* Real-world pitch coordinate transformation

---

# 👨‍💻 Author

**Babar Khan Durrani**

BS Robotics & Intelligent Systems
Bahria University Islamabad

---

# ⭐ If You Found This Project Interesting

Consider giving the repository a **star ⭐**.
