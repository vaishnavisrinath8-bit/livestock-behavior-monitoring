# 🐄 AI-Based Real-Time Livestock Behavior & Inactivity Monitoring System

### Intelligent Animal Behavior and Welfare Monitoring Using Computer Vision

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Object%20Detection-purple)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![SQLite](https://img.shields.io/badge/Database-SQLite-blue)

## 📌 Project Overview

The **AI-Based Real-Time Livestock Behavior & Inactivity Monitoring System** is a computer vision-based application designed to monitor livestock activity and identify prolonged periods of inactivity.

The system uses artificial intelligence and video processing techniques to detect cows, track individual animals, recognize observable behaviors, and monitor their activity durations.

It aims to help farmers and livestock managers observe animal activity more efficiently without requiring wearable sensors.

The application processes uploaded videos or webcam footage and presents monitoring information through an interactive Streamlit dashboard.

## 🚨 Problem Statement

Monitoring every animal continuously on a large livestock farm is difficult and time-consuming.

Farmers may not immediately notice changes in animal activity, feeding patterns, or resting behavior. Prolonged unusual inactivity may indicate a potential animal welfare concern.

Traditional monitoring methods often depend on manual observation and may not provide continuous activity records.

## 💡 Proposed Solution

This project proposes an AI-powered livestock monitoring application that uses computer vision to:

- Detect cows from video footage.
- Track individual animals using object tracking.
- Classify observable behaviors such as Standing, Feeding, and Lying/Resting.
- Calculate individual animal activity durations.
- Monitor prolonged lying or inactivity.
- Generate configurable observation alerts.
- Store activity records and alerts in an SQLite database.
- Display monitoring information through an interactive dashboard.

**Note:** Prolonged lying or inactivity is treated as an observation alert, not a medical diagnosis.

## 🎯 Project Objectives

1. Automate livestock detection using computer vision.
2. Track individual cows in video footage.
3. Identify observable livestock behaviors.
4. Monitor individual animal activity durations.
5. Generate alerts when configured inactivity thresholds are exceeded.
6. Maintain historical activity records.
7. Provide a user-friendly livestock monitoring dashboard.
8. Reduce the need for continuous manual observation.

## ✨ Key Features

### 1. Cow Detection
- YOLOv8n-based cow detection.
- Bounding box visualization.
- Detection confidence scores.
- Frame-by-frame video processing.

### 2. Individual Animal Tracking
- ByteTrack-based object tracking.
- Persistent tracking IDs across frames where possible.
- Individual animal identification.
- Tracking visualization.

### 3. Behavior Classification
The application is designed to recognize three main behaviors:

- Standing
- Feeding
- Lying/Resting

It supports an optional custom-trained behavior model.

A baseline estimation mode will be available when a trained behavior model is not installed. Uncertain observations will be labelled Unknown rather than presented as reliable predictions.

### 4. Inactivity Monitoring
- Individual animal activity tracking.
- Continuous lying-duration calculation.
- Configurable inactivity thresholds.
- Temporary tracking-loss handling.
- Activity duration monitoring.

### 5. Alert Generation
- Configurable inactivity alerts.
- Warning and critical alert levels.
- Alert cooldown mechanism.
- Alert history.
- Automatic alert resolution when the monitored condition ends.

### 6. Interactive Dashboard
The Streamlit dashboard will provide:

- Total animals detected.
- Current behavior information.
- Animal tracking details.
- Behavior distribution charts.
- Inactivity duration charts.
- Recent alerts.
- Historical activity records.

### 7. Database Management
SQLite will store:

- Animal information.
- Activity logs.
- Inactivity alerts.
- Monitoring history.

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Computer Vision | OpenCV, NumPy, Pillow |
| Object Detection | YOLOv8n |
| Deep Learning | PyTorch |
| Object Tracking | ByteTrack |
| Frontend | Streamlit |
| Data Visualization | Plotly |
| Data Processing | Pandas |
| Database | SQLite |
| Testing | Pytest |
| Version Control | Git and GitHub |

## 🏗️ System Architecture

The application follows a modular architecture.

```text
              VIDEO INPUT
                   |
          Uploaded Video / Webcam
                   |
                   v
           IMAGE PREPROCESSING
                   |
                   v
             COW DETECTION
                YOLOv8
                   |
                   v
          INDIVIDUAL TRACKING
               ByteTrack
                   |
                   v
          BEHAVIOR CLASSIFICATION
                   |
          ---------------------
          |         |         |
       Standing  Feeding  Lying/Resting
          |         |         |
          ---------------------
                   |
                   v
          INACTIVITY MONITORING
                   |
                   v
            ALERT GENERATION
                   |
                   v
             SQLITE DATABASE
                   |
                   v
          STREAMLIT DASHBOARD
                   |
                   v
          ANALYTICS AND REPORTS
```

## 📂 Project Structure

```text
livestock-behavior-monitoring/
│
├── app.py
├── config.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── models/
│   ├── cow_detector.py
│   ├── behavior_classifier.py
│   └── tracker.py
│
├── processing/
│   ├── video_processor.py
│   ├── frame_processor.py
│   ├── inactivity_engine.py
│   └── alert_engine.py
│
├── database/
│   ├── db.py
│   └── database_operations.py
│
├── dashboard/
│   ├── sidebar.py
│   ├── overview.py
│   ├── animal_details.py
│   ├── analytics.py
│   └── alerts.py
│
├── utils/
│   ├── image_utils.py
│   ├── video_utils.py
│   └── time_utils.py
│
├── data/
│   ├── uploads/
│   ├── processed/
│   ├── sample_videos/
│   └── livestock.db
│
├── weights/
│   └── README.md
│
├── tests/
│   ├── test_database.py
│   ├── test_inactivity.py
│   └── test_processing.py
│
└── assets/
    └── logo.png
```

## ⚙️ Installation and Setup

### Prerequisites

Install the following:

- Python 3.11 or 3.12 recommended.
- Git.
- Visual Studio Code or Google Antigravity.
- A Windows computer.
- Internet connection for initial dependency and model downloads.

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/livestock-behavior-monitoring.git
```

### Step 2: Navigate to the Project Folder

```bash
cd livestock-behavior-monitoring
```

### Step 3: Create a Virtual Environment

Windows PowerShell:

```powershell
py -3.11 -m venv venv
```

### Step 4: Activate the Virtual Environment

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 5: Install Dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 6: Run the Application

```powershell
streamlit run app.py
```

The application will open in your default browser.

## 🎥 How to Use the Application

1. Launch the Streamlit application.
2. Open the Dashboard Overview.
3. Navigate to Live Monitoring.
4. Upload a supported livestock video.
5. Start video processing.
6. View cow detection and tracking results.
7. Observe available behavior classifications.
8. Monitor individual animal activity durations.
9. Configure the inactivity threshold.
10. View generated alerts and activity analytics.

Supported video formats:

- MP4
- AVI
- MOV

Webcam monitoring will be available as an optional input mode.

## 🧠 AI Model Information

### Cow Detection

The project initially uses the pretrained YOLOv8n model to detect cows.

The pretrained model provides object detection, not livestock behavior recognition.

### Behavior Classification

A separate custom-trained model is required for reliable recognition of:

- Standing
- Feeding
- Lying/Resting

The architecture supports integrating this model without redesigning the entire application.

### Object Tracking

ByteTrack is used to associate detections across consecutive video frames.

Tracking IDs may change when animals disappear for extended periods or overlap.

## 🗄️ Database

The application uses SQLite for local data storage.

Planned database tables:

**Animals**

Stores animal tracking information and observation status.

**Activity Logs**

Stores observed animal behaviors and their durations.

**Alerts**

Stores inactivity alerts, severity levels, timestamps, and resolution status.

## 🧪 Testing

The project uses Pytest for automated testing.

Run tests using:

```powershell
pytest -v
```

Tests cover:

- Database initialization.
- Animal record insertion.
- Activity log insertion.
- Alert creation.
- Inactivity duration calculations.
- Timer reset conditions.
- Alert cooldown.
- Invalid video handling.
- Missing behavior model handling.

## ⚠️ Limitations

- Reliable behavior classification requires a suitable annotated livestock dataset and a trained custom model.
- Detection accuracy depends on video quality, camera angle, lighting, and animal visibility.
- Tracking IDs may change when animals overlap or disappear for extended periods.
- Baseline behavior estimation may be uncertain.
- Prolonged lying is not necessarily a sign of illness.
- The system is an observation and monitoring tool, not a veterinary diagnostic system.

## 🚀 Future Enhancements

- Train a custom livestock behavior recognition model.
- Improve individual cow identification.
- Add feeding and drinking activity analysis.
- Support multiple livestock species.
- Introduce farm-level monitoring.
- Add historical activity reports.
- Improve real-time monitoring performance.
- Explore integration with farm management systems.

## 👩‍💻 Development Status

**Project Status: Under Development**

The application is being developed incrementally, beginning with cow detection and tracking, followed by behavior classification, inactivity monitoring, database integration, and dashboard development.

Features will be marked as completed only after implementation and testing.

## 📜 License

This project is intended for educational and research purposes.

An MIT License may be added to the repository if the project is distributed under that license.

---

**Built with Python, Artificial Intelligence, and Computer Vision.** 🐄
