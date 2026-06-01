

# Edge-Based Smart Suspension System using AI Object Detection

> A real-time pothole and speed bump detection system that uses a mobile camera, YOLOv8 AI model, and a laptop edge server to predictively control vehicle suspension — a cost-effective alternative to BMW's onboard Road Preview system.

---

## Project Overview

This project is focused on building an intelligent vehicle suspension system using computer vision and edge computing.

**The core idea:** instead of mounting expensive AI hardware inside the car (like BMW's ₹5–8 lakh Road Preview system), a camera streams road footage to a nearby **edge server**. The server runs a **YOLOv8 object detection model** to detect potholes and speed bumps ahead of the vehicle. Detection results — including distance, severity, and required suspension adjustment — are sent back in real time, giving the suspension controller time to react **before** the wheel hits the obstacle.


## Project Goals

- Detect **potholes** and **speed bumps** in real time from a forward-facing camera
- Estimate the **distance** to each obstacle using the pinhole camera model
- Classify **severity** (LOW / MEDIUM / HIGH) based on bounding box area
- Calculate the **exact timing** to fire suspension actuators based on vehicle speed
- Stream annotated video back to the phone for **live visualization**
- Demonstrate the complete edge computing pipeline as a working prototype

---

## System Architecture

```
Camera
      │
      │  HTTP/RTSP stream
      ▼
Laptop (Edge Server)
      │
      ├── OpenCV reads frames
      ├── YOLOv8 runs inference
      ├── Distance estimation
      ├── Severity classification
      ├── Timing calculation (dist / speed)
      └── FGenerate command JSON
      │
      ▼
Suspension Controller
```

### Data Flow

```
Frame captured
      │
      ▼
YOLOv8 Inference
      │
      ├── Bounding box (x1, y1, x2, y2)
      ├── Class label  (pothole / speed_bump)
      └── Confidence   (0.0 – 1.0)
      │
      ▼
Distance Estimation
      │   distance_m = (real_height × focal_px) / pixel_height
      │
      ▼
Severity Classification
      │   Area > 25000 px² → HIGH
      │   Area > 10000 px² → MEDIUM
      │   Area < 10000 px² → LOW
      │
      ▼
Trigger Timing
      │   wait_ms = (distance_m / speed_ms × 1000) − 80
      │
      ▼
Command JSON → Suspension Controller
```

---

## Dataset Details

<table>
  <tr>
    <td>Classes</td>
    <td><code>pothole: 0</code>, <code>bump: 1</code></td>
  </tr>
  <tr>
    <td>mAP@50</td>
    <td>70%</td>
  </tr>
  <tr>
    <td>Epochs Trained</td>
    <td>30</td>
  </tr>
</table>


### Pothole and Bumps Dataset:
```bash 
https://universe.roboflow.com/jampanis-workspace/potholes-bumps-hnwyb/dataset/1
```

---

## Project Structure

```
SMART-SUSPENSION/
│
├── Potholes & Bumps/          # Dataset (from Roboflow)
│   ├── train/                
│   ├── valid/                
│   ├── test/                  
│   └── data.yaml             
│
├── runs/
│   └── detect/
│       └── training_outputs/
│           └── suspension_v1/
│               ├── weights/
│               │   ├── best.pt   # Trained model
│               │   └── last.pt 
│               ├── args.yaml    
│               └── results.csv
│
├── setup/ 
│   ├── download_dataset.py
│   ├── setup.bash             # Linux/Mac setup
│   ├── setup.bat              # Windows setup
│   └── setup.md               # Setup instructions
│
├── camera.py                  # Live camera inference script
├── imagedetect.py             # Run detection on a single image
├── videodetect.py             # Run detection on a video file
├── train.py                   # Model training script
├── potholedetection.ipynb     # Jupyter notebook (experiments)
│
├── output.mp4                 # Detection output video (sample)
├── output2.mp4                # Detection output video (sample 2)
├── test.mp4                   # Test video input
├── test2.mp4                  # Test video input 2
│
├── .gitignore
└── README.md

```

## Setup & Installation

 Refer to [setup.md](setup\setup.md)


### Clone the repo

```bash
git clone https://github.com/jampanirohith/Edge-Based-Smart-Suspension
```

---

### Suspension Actions

| Detection | Action | Description |
|---|---|---|
| `pothole` | **DROP** | Extend suspension to absorb the drop — chassis stays level |
| `speed_bump` | **RAISE** | Raise all 4 wheels + soften damping before impact |
