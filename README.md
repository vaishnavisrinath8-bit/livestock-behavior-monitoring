# 🐄 AI-Based Real-Time Livestock Behavior & Inactivity Monitoring System

### Intelligent Animal Behavior and Welfare Monitoring Using Computer Vision

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Object%20Detection-purple)
![ByteTrack](https://img.shields.io/badge/Tracking-ByteTrack-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![SQLite](https://img.shields.io/badge/Database-SQLite%20WAL-blue)
![Tests](https://img.shields.io/badge/Tests-21%2F21%20Passing-brightgreen)

---

## 📌 Project Overview

The **AI-Based Real-Time Livestock Behavior & Inactivity Monitoring System** is a computer vision-based application designed to monitor livestock activity, recognize behaviors, and identify prolonged periods of inactivity in cattle herds.

The system uses artificial intelligence and video processing to detect cows, track individual animals across video frames, estimate observable behaviors, track activity durations, and generate configurable observation alerts when an animal remains in a resting/lying state beyond safe thresholds.

The application operates locally without requiring expensive cloud subscriptions or invasive wearable sensors. It processes uploaded video files or live webcam surveillance feeds and presents real-time observations and historical analytics through an interactive Streamlit dashboard.

---

## 🚨 Problem Statement

Continuous visual observation of every animal in large commercial livestock barns or pastures is labor-intensive and error-prone. 

Subtle changes in activity, feeding frequency, or resting patterns can easily go unnoticed. Prolonged inactivity or recumbency may serve as an early indicator of lameness, illness, calving difficulty, or animal distress.

Traditional monitoring methods depend largely on intermittent manual checks or costly wearable tags (such as collars and ear sensors) that require regular maintenance, battery replacement, and physical contact with animals.

---

## 💡 Proposed Solution

This project provides a non-invasive, vision-based livestock monitoring solution that:

1. **Detects cows** in video streams using a lightweight YOLOv8n object detection model.
2. **Tracks individual cattle** using ByteTrack, maintaining consistent tracking IDs across consecutive frames.
3. **Classifies observable behaviors** into four standard classes: *Standing*, *Feeding*, *Lying/Resting*, and *Unknown*.
4. **Calculates continuous inactivity** durations specifically during prolonged lying/resting postures.
5. **Generates configurable observation alerts** at Warning and Critical thresholds with rate-limited cooldowns.
6. **Persists data in an SQLite database** with transaction safety and WAL concurrency.
7. **Presents analytical charts and live feeds** through an interactive Streamlit web dashboard.

> [!NOTE]
> **Important Welfare Disclaimer:** Prolonged lying or inactivity is treated strictly as an **observation alert** for farm staff inspection, not as an automated veterinary or medical diagnosis.

---

## 🎯 Project Objectives

1. Automate cattle detection using deep learning computer vision.
2. Maintain persistent individual tracking IDs across video frames.
3. Classify observable postures (Standing, Feeding, Lying/Resting).
4. Monitor continuous individual inactivity timers using video playback timestamps (independent of hardware processing speed).
5. Generate tiered observation alerts (Normal, Warning, Critical) when inactivity thresholds are exceeded.
6. Enforce alert cooldown periods to prevent repeated per-frame notification spam.
7. Maintain historical activity logs and alert registries in an SQLite database.
8. Deliver an intuitive dashboard for farm managers and veterinary observers.
9. Provide seamless support for integrating custom-trained behavior models.

---

## ✨ Key Implemented Features

### 1. Cow Detection (YOLOv8n)
- Pretrained lightweight `yolov8n.pt` model.
- Class filtering specifically targeting COCO Class `19` (`cow`).
- Rejection of unrelated objects (humans, vehicles, barn equipment).
- Bounding box rendering with confidence chips and alert banners.

### 2. Multi-Object Animal Tracking (ByteTrack)
- Ultralytics integrated ByteTrack association algorithm.
- Bounding box trajectory association across consecutive frames.
- Identification chips: `Cow #1`, `Cow #2`, etc.
- Grace period handling to absorb momentary visual occlusions without dropping state.

### 3. Dual-Mode Behavior Classification
- **Mode A (Custom AI Model):** Direct loading and inference of fine-tuned weights from `weights/behavior_model.pt`.
- **Mode B (Baseline Estimation Mode):** Transparent, explainable posture estimation based on feeding bunk zones and bounding box geometric aspect ratios ($AR = \text{width} / \text{height}$). Cases with ambiguous postures are explicitly assigned **`Unknown`** rather than fabricating false confidence scores.

### 4. Inactivity Monitoring Engine
- Tracks continuous lying duration per individual animal.
- Timestamp calculation based on video FPS and frame indexes ($t = \text{frame} / \text{fps}$), avoiding CPU clock drift.
- Immediate timer reset when an animal transitions from resting to standing or feeding.
- Missing detections during tracking loss do **not** falsely inflate continuous lying duration.

### 5. Multi-Tiered Alert Engine
- **Normal:** Animal active or resting within normal time limits.
- **Warning:** Animal resting continuously past 70% of the configured threshold.
- **Critical:** Animal resting continuously past 100% of the threshold.
- Automatic alert resolution when the animal stands up or condition normalizes.
- Configurable alert cooldown to eliminate duplicate database records.

### 6. Interactive Streamlit Dashboard
- **Dashboard Overview:** Live database KPIs, behavior distribution donut charts, and alert tables.
- **Live Monitoring:** Real-time video playback with bounding boxes, state tables, and progress controls.
- **Animal Tracking:** Complete registry of all active and previously seen animals.
- **Animal Details:** Individual animal welfare profiles with cumulative behavior breakdowns.
- **Behavior Analytics:** Population-level duration distributions and event timelines.
- **Inactivity Alerts:** Filterable alert history with manual and automated resolution status.
- **Activity History:** Chronological behavior transition events.
- **Settings:** Configurable confidence thresholds, demo mode (10s threshold) vs standard mode (15 min), frame skip intervals, and feeding bunk coordinates.

---

## 🛠️ Technology Stack

| Component | Technology | Version | Purpose |
|---|---|---|---|
| **Language** | Python | 3.11 (64-bit) | High compatibility with PyTorch Windows binaries |
| **Computer Vision** | OpenCV, NumPy, Pillow | 4.11+, 1.26+, 13.0+ | Frame decoding, resizing, drawing, geometry |
| **Object Detection** | Ultralytics YOLOv8 | 8.4+ | COCO cow detection |
| **Tracking** | ByteTrack | Built-in | Multi-object tracking across frames |
| **Deep Learning** | PyTorch, Torchvision | 2.15+, 0.29+ | Tensor execution and model inference |
| **Web Dashboard** | Streamlit | 1.67+ | Reactive web user interface |
| **Visualization** | Plotly | 7.1+ | Interactive charts (Donut, Bar, Scatter) |
| **Data Processing**| Pandas | 3.0+ | Tabular manipulation and aggregation |
| **Database** | SQLite (WAL mode) | 3 | Thread-safe, persistent local relational storage |
| **Testing** | Pytest | 9.2+ | Automated unit and integration test suite |

---

## 🏗️ System Architecture

```text
                     +---------------------------------------+
                     |             VIDEO INPUT               |
                     |   (Uploaded MP4/AVI/MOV or Webcam)    |
                     +---------------------------------------+
                                         |
                                         v
                     +---------------------------------------+
                     |         IMAGE PREPROCESSING           |
                     |  (Aspect Resize, RGB/BGR, Frame Skip) |
                     +---------------------------------------+
                                         |
                                         v
                     +---------------------------------------+
                     |           COW DETECTION               |
                     |      YOLOv8n (COCO Class 19)          |
                     +---------------------------------------+
                                         |
                                         v
                     +---------------------------------------+
                     |        INDIVIDUAL TRACKING            |
                     |       ByteTrack Association          |
                     +---------------------------------------+
                                         |
                                         v
                     +---------------------------------------+
                     |        BEHAVIOR CLASSIFICATION        |
                     |  Mode A: Custom weights (behavior.pt) |
                     |  Mode B: Explainable Baseline Mode    |
                     |  [Standing | Feeding | Lying | Unknown|
                     +---------------------------------------+
                                         |
                                         v
                     +---------------------------------------+
                     |         INACTIVITY ENGINE             |
                     |   - Continuous Lying Duration Timer   |
                     |   - Grace Period Loss Absorption      |
                     |   - Timer Reset on Posture Change     |
                     +---------------------------------------+
                                         |
                                         v
                     +---------------------------------------+
                     |           ALERT ENGINE                |
                     |   - Normal / Warning / Critical       |
                     |   - Cooldown & Duplicate Prevention   |
                     |   - Automatic Condition Resolution    |
                     +---------------------------------------+
                                         |
                     +-------------------+-------------------+
                     |                                       |
                     v                                       v
+---------------------------------------+   +---------------------------------------+
|           SQLITE DATABASE             |   |         STREAMLIT DASHBOARD           |
|  - animals (Tracking, Times, Status)  |   |  - Live Annotated Feed Display        |
|  - activity_logs (Interval Durations) |   |  - Real-time KPI Metric Cards         |
|  - alerts (Active / Resolved Events)  |   |  - Plotly Behavioral Analytics        |
+---------------------------------------+   +---------------------------------------+
```

---

## 📂 Project Folder Structure

```text
livestock-behavior-monitoring/
├── app.py                     # Main Streamlit web application entry point
├── config.py                  # System constants, color schemes, and configuration
├── conftest.py                # Global test configuration and sys.path binding
├── pytest.ini                 # Pytest configuration file
├── requirements.txt           # Pinned production dependencies
├── verify_pipeline.py         # End-to-end headless pipeline validation script
├── README.md                  # Comprehensive project documentation
├── .gitignore                 # Git ignore file for Python, models, and databases
│
├── models/
│   ├── cow_detector.py        # YOLOv8n detector with COCO cow class filtering
│   ├── tracker.py             # ByteTrack multi-object tracking manager
│   └── behavior_classifier.py # Modular behavior classification (Custom AI / Baseline)
│
├── processing/
│   ├── frame_processor.py     # 12-step single frame execution pipeline
│   ├── video_processor.py     # Stream reading, FPS timestamp calculation, webcam loop
│   ├── inactivity_engine.py   # Continuous lying timers, state updates, grace periods
│   └── alert_engine.py        # Alert thresholds, cooldowns, and auto-resolutions
│
├── database/
│   ├── db.py                  # Thread-safe SQLite connection and table initialization
│   └── database_operations.py # Parameterized queries, upserts, logs, KPIs, alerts
│
├── dashboard/
│   ├── sidebar.py             # Global branding, system status chip, and navigation
│   ├── overview.py            # Overview page with KPI cards and Plotly charts
│   ├── live_monitoring.py     # Live video upload, webcam feed, and real-time bounding boxes
│   ├── animal_details.py      # Individual animal registry and welfare history profiles
│   ├── analytics.py           # Population behavioral distributions and timelines
│   ├── alerts.py              # Inactivity alerts table with manual resolution options
│   ├── activity_history.py    # Chronological behavior event logs
│   └── settings.py            # Threshold adjustments, demo mode toggle, ROI setup
│
├── utils/
│   ├── image_utils.py         # Resizing, color conversions, box & badge drawing
│   ├── video_utils.py         # Video metadata extraction and file integrity validation
│   ├── time_utils.py          # Duration formatting (HMS) and video timestamp logic
│   ├── validators.py          # Range and format checks for user inputs
│   └── generate_sample_video.py # Synthetic & photographic video generation utility
│
├── data/
│   ├── uploads/               # User-uploaded video footage
│   ├── processed/             # Output files and exported frames
│   ├── sample_videos/         # Included demonstration videos (demo_cattle.mp4)
│   └── livestock.db           # SQLite database file (WAL mode)
│
├── weights/
│   ├── README.md              # Instructions for downloading and placing model weights
│   └── yolov8n.pt             # Pretrained YOLOv8n object detection weights
│
└── tests/
    ├── test_database.py       # Tests for table creation, upserts, logs, KPIs, alerts
    ├── test_detection.py      # Tests for cow detection filtering and dataclasses
    ├── test_tracking.py       # Tests for ByteTrack association and reset behavior
    ├── test_behavior.py       # Tests for baseline heuristics, ROI, and fallback modes
    ├── test_inactivity.py     # Tests for lying duration timer and reset conditions
    ├── test_alerts.py         # Tests for alert cooldown, escalation, and auto-resolution
    └── test_processing.py     # Tests for video validation and full pipeline execution
```

---

## ⚙️ Installation and Setup

### Prerequisites

- **Operating System:** Windows 10/11 (64-bit)
- **Python:** Python 3.11 (64-bit recommended for seamless PyTorch binary support)
- **Git**

### Step 1: Open PowerShell in the Project Directory

```powershell
cd C:\Users\vaish\livestock-behavior-monitoring
```

### Step 2: Create a Dedicated Python 3.11 Virtual Environment

```powershell
py -3.11 -m venv venv
```

### Step 3: Activate the Virtual Environment

```powershell
.\venv\Scripts\Activate.ps1
```

*(If PowerShell script execution is restricted on your system, enable it using `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned`)*

### Step 4: Upgrade Pip and Install Dependencies

```powershell
.\venv\Scripts\python.exe -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Running the Application

### Launch the Streamlit Dashboard

```powershell
streamlit run app.py
```

The application will start the local web server and automatically open the monitoring center in your default browser at:
`http://localhost:8501`

### Running the Verification Script

To verify all components (models, tracking, database, timers) headlessly without opening the browser:

```powershell
python verify_pipeline.py
```

---

## 🧪 Automated Testing

The project includes 21 automated unit and integration tests covering all critical operational requirements.

To run the complete test suite:

```powershell
pytest -v
```

### Test Coverage Summary:
- `tests/test_database.py`: Database schema initialization, animal upserts, activity logs, alert insertion/resolution, and live KPI calculations.
- `tests/test_detection.py`: YOLOv8 cow class filtering, bounding box coordinates, confidence formatting, and empty frame safety.
- `tests/test_tracking.py`: ByteTrack ID assignment, active track persistence, and stream resets.
- `tests/test_behavior.py`: Graceful fallback when custom weights are absent, feeding bunk inclusion, and aspect ratio posture heuristics.
- `tests/test_inactivity.py`: Continuous lying duration accumulation, immediate timer resets on posture change, and grace period handling.
- `tests/test_alerts.py`: Warning/Critical threshold triggering, cooldown duplicate prevention, and automatic alert resolution.
- `tests/test_processing.py`: Rejection of corrupted/empty files and end-to-end frame processing.

---

## 🧠 Behavior Classification & Custom Model Training Guide

The system is architected to recognize four behavioral states:
1. **Standing**
2. **Feeding**
3. **Lying/Resting**
4. **Unknown**

### How the Dual-Mode Classifier Works
- **Custom AI Mode:** When a fine-tuned YOLO model is placed at `weights/behavior_model.pt`, the application automatically loads it and uses its predictions and confidence scores.
- **Baseline Estimation Mode:** When no custom weights exist, the system operates in an explainable heuristic mode:
  - If a cow's feet are within the user-defined feeding bunk zone $\rightarrow$ **Feeding**.
  - If the bounding box aspect ratio $AR = \text{width} / \text{height} \ge 1.75$ $\rightarrow$ **Lying/Resting**.
  - If $AR \le 1.20$ $\rightarrow$ **Standing**.
  - Ambiguous cases are marked **Unknown** (no artificial confidence scores are fabricated).

### Step-by-Step Custom Training Instructions

To train a custom YOLO behavior classification or detection model for your farm:

#### 1. Data Collection
- Record pen or pasture video footage across diverse lighting conditions (morning, afternoon, indoor artificial lighting).
- Ensure cattle are visible from side and angled perspectives.

#### 2. Frame Extraction
Extract frames from video recordings at regular intervals using OpenCV:
```python
import cv2
cap = cv2.VideoCapture("barn_footage.mp4")
idx, saved = 0, 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    if idx % 30 == 0:  # save every 30th frame (1 frame per second at 30 fps)
        cv2.imwrite(f"dataset/raw_frames/frame_{saved:05d}.jpg", frame)
        saved += 1
    idx += 1
cap.release()
```

#### 3. Annotation
- Use open-source annotation tools such as **Roboflow**, **CVAT**, or **LabelImg**.
- Label cattle bounding boxes with one of three classes:
  - `0: Standing`
  - `1: Feeding`
  - `2: Lying`
- Split the dataset into 80% Training and 20% Validation sets.

#### 4. Training with YOLOv8
Create a `behavior_data.yaml` configuration:
```yaml
path: ../dataset
train: images/train
val: images/val
names:
  0: Standing
  1: Feeding
  2: Lying
```

Train using Ultralytics in your virtual environment:
```powershell
yolo detect train data=behavior_data.yaml model=yolov8n.pt epochs=50 imgsz=640 batch=16
```

#### 5. Integration
Copy the trained `best.pt` file into the weights folder:
```powershell
Copy-Item runs/detect/train/weights/best.pt weights/behavior_model.pt
```
Restart `app.py`. The application will immediately switch to **Custom AI Mode**!

---

## ⏱️ Inactivity Threshold & Demonstration Mode

- **Standard Operational Mode:** Configured to **15 minutes** (900 seconds) by default. Suitable for commercial farm production where lying is normal, but prolonged recumbency exceeding 15–30 minutes warrants inspection.
- **Demonstration Mode (College Lab / Review):**
  - Toggle **Demonstration Mode** in the **Settings** tab.
  - Reduces the inactivity threshold to **10 seconds**.
  - Allows immediate demonstration of:
    1. Continuous lying accumulation.
    2. Warning alert at 7.0 seconds (70%).
    3. Critical alert at 10.0 seconds (100%).
    4. Real-time visual alert banner on the video stream.
    5. Automatic resolution as soon as the animal stands up.

---

## 🗄️ Database Schema (SQLite)

The SQLite database (`data/livestock.db`) uses Write-Ahead Logging (`WAL` mode) for concurrent read access from the Streamlit UI during continuous video processing.

### Tables:

1. **`animals`**
   - `id`: Integer primary key.
   - `tracking_id`: Unique ByteTrack integer assigned to the cow.
   - `first_seen`: ISO 8601 UTC timestamp of first detection.
   - `last_seen`: ISO 8601 UTC timestamp of most recent detection.
   - `current_behavior`: Current observable behavior.
   - `total_observed_seconds`: Real video duration the animal was tracked.
   - `status`: `Active`, `GracePeriod`, or `Lost`.

2. **`activity_logs`**
   - `id`: Integer primary key.
   - `tracking_id`: Foreign key reference to tracked cow.
   - `behavior`: Completed behavior state (*Standing*, *Feeding*, *Lying*).
   - `duration_seconds`: Continuous seconds spent in that behavior before transitioning.
   - `timestamp`: UTC timestamp when interval completed.
   - `source_type`: `video` or `webcam`.

3. **`alerts`**
   - `id`: Integer primary key.
   - `tracking_id`: Foreign key reference to tracked cow.
   - `alert_type`: `Prolonged Inactivity`.
   - `inactivity_seconds`: Duration of continuous lying observed.
   - `severity`: `Warning` or `Critical`.
   - `message`: Human-readable observation alert message.
   - `created_at`: UTC timestamp of alert generation.
   - `resolved`: `0` for active, `1` when condition resolves.

---

## ⚠️ Limitations & Ethical Considerations

1. **Not a Veterinary Diagnosis:** Prolonged lying down is a common and normal resting behavior for cattle. Alerts generated by this application are welfare observation prompts, not medical diagnoses of disease.
2. **Camera Perspective:** Aspect-ratio heuristics in Baseline Mode depend on roughly horizontal or slightly elevated camera angles. Overhead bird's-eye views require a custom-trained keypoint or segmentation model.
3. **ID Switching:** ByteTrack tracking IDs may change when animals overlap extensively, undergo severe occlusion, or leave and re-enter the camera frame.
4. **Lighting Conditions:** Severe shadows or nocturnal footage without infrared illumination will reduce detection confidence.

---

## 🚀 Future Enhancements

- Integrate keypoint pose estimation (YOLOv8-pose) to track head and leg positions for drinking behavior.
- Support multi-camera pen networks with global Re-ID across camera handoffs.
- Add automated email/webhook notifications for farm management software integration.
- Deploy edge optimization with ONNX Runtime or TensorRT for low-power edge gateways.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
