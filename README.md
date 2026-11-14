Autonomous Retail Intelligence System

An end-to-end computer vision and predictive analytics platform designed for retail stores. The system tracks customer footfall, dwell time, and spatial movement in real time using YOLOv8 and DeepSORT, logs behavioral metrics into SQLite databases, and translates raw video feeds into actionable business insights via an interactive Streamlit and FastAPI executive dashboard.

System Architecture & Block Diagram

The system operates across four sequential stages—from raw optical video ingestion to executive business intelligence:

+-----------------------------------------------------------------------------------+
| 1. VIDEO INGESTION & EDGE HARDWARE                                                |
|    [IP CCTV / RTSP Stream / Local MP4 Video]                                      |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| 2. COMPUTER VISION & TRACKING ENGINE (vision_pipeline.py)                         |
|    [YOLOv8n Person Detection] ---> [DeepSORT Multi-Object Tracking]               |
|               |                                     |                             |
|               v                                     v                             |
|    [Gaussian Spatial Heatmap]      [Zone Dwell-Time & Footfall Logger]            |
|        (heatmap_output.png)                  (zones.json calibration)             |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| 3. DATA STORAGE & SYNTHETIC SIMULATION LAYER                                      |
|    [video_data.db (Live Logs)]  <--->  [generator.py / retail_data.db (75d Mock)] |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| 4. ANALYTICS, API & EXECUTIVE DASHBOARD (api.py & dashboard.py)                   |
|    +---------------------------+  +------------------------+  +----------------+  |
|    | FastAPI REST Endpoints    |  | ML & Optimization Lab  |  | Gemini AI      |  |
|    | - Telemetry & Zone Stats  |  | - Purchase Prediction  |  | Retail Advisor |  |
|    +---------------------------+  | - Knapsack Budget Opt. |  +----------------+  |
|                                   | - 7-Day Traffic & A/B  |                      |
|                                   +------------------------+                      |
+-----------------------------------------------------------------------------------+


Block Diagram Workflow Explanation

Video Ingestion: Raw footage is captured either via live RTSP IP CCTV camera feeds or uploaded MP4 test files.

Computer Vision & Tracking Engine (vision_pipeline.py):

Every video frame passes through YOLOv8n to detect human bounding boxes.

DeepSORT assigns persistent tracking IDs across consecutive frames so individual shoppers are not double-counted.

Foot coordinates are checked against polygon boundaries defined in zones.json to calculate exact dwell times, while accumulated coordinates generate heatmap_output.png.

Data Storage & Simulation (video_data.db & generator.py):

Live visitor sessions exceeding the minimum dwell threshold are committed to SQLite (video_data.db).

For multi-week forecasting, back-testing, and model training, generator.py supplies realistic 75-day retail behavioral datasets (retail_data.db).

API & Executive Dashboard (api.py & dashboard.py):

api.py exposes structured REST endpoints for external integrations and telemetry queries.

dashboard.py renders the 8-tab Streamlit command center—running Gradient Boosting purchase classifiers, multiple-choice knapsack budget optimization, time-series staffing forecasts, K-Means customer segmentation, and Google Gemini AI strategic recommendations.

Key Features

Real-Time Multi-Object Tracking: Detects and tracks unique shoppers across customizable store zones (Entrance Corridor, Main Display Shelf Zone, Checkout Counter) using YOLOv8 and DeepSORT.

Spatial Movement Heatmaps: Automatically generates Gaussian movement heatmaps (heatmap_output.png) highlighting high-traffic corridors and engagement hotspots.

Predictive Purchase Modeling: Evaluates visitor purchase probability using a Gradient Boosting classifier trained on dwell time, zone activity, and demographic features.

Revenue & Budget Optimization: Uses dynamic programming (multiple-choice knapsack solver) to recommend the highest-ROI marketing and store-layout interventions within a fixed budget.

7-Day Footfall Forecasting & Staffing: Back-tests time-series forecasting models (Seasonal Average vs. Ridge Regression) to predict upcoming store traffic and calculate optimal hourly staff schedules.

A/B Test Planner & Customer Segmentation: Uses K-Means clustering to segment shoppers by intent and calculates statistically valid sample sizes for in-store promotional experiments.

AI Retail Advisor: Integrates Google Gemini to provide natural-language executive summaries, multi-period comparisons, and interactive Q&A.

Codebase & Module Breakdown

Below is a detailed breakdown of what each core Python module in the repository analyzes and executes:

vision_pipeline.py (Computer Vision & Spatial Telemetry)

What it analyzes: Processes store CCTV video feeds frame by frame using YOLOv8 for person detection and DeepSORT for multi-object tracking. It evaluates shopper foot positions against custom spatial boundaries (zones.json), measures exact dwell times per zone, renders Gaussian movement heatmaps, and writes verified visitor session logs into video_data.db.

dashboard.py (Executive Analytics & ML Interface)

What it analyzes: Powers the interactive 8-tab Streamlit user interface. It ingests either live camera logs (video_data.db) or 75-day historical datasets to analyze traffic patterns, demographic distributions, and zone dwell metrics. It executes Gradient Boosting purchase probability predictions, runs dynamic programming (knapsack) budget optimizations, performs K-Means customer clustering, evaluates model ROC-AUC/calibration metrics, and connects with Google Gemini for AI-driven store recommendations.

api.py (Backend Telemetry & REST Service)

What it analyzes: Manages the FastAPI backend server and RESTful endpoints. It queries and aggregates store telemetry, zone performance metrics, visitor logs, and forecasting outputs from the underlying SQLite databases to serve clean JSON responses for external clients and microservices.

generator.py (Synthetic Retail Data & Behavioral Simulator)

What it analyzes: Models and generates realistic multi-week retail datasets (covering 75 days of customer footfall, age/gender demographics, zone dwell times, promotional lift, and purchase outcomes). This provides a statistically rich baseline for training machine learning classifiers, running 14-day forecasting back-tests, and demonstrating full-scale dashboard analytics when only short CCTV clips are available.

Hardware Architecture (Edge & Deployment Setup)

While the system can run locally on a standard laptop using pre-recorded video files, deploying it in a live physical retail environment requires the following hardware components:

1. Edge Compute & Processing Unit

Primary Edge Node: NVIDIA Jetson Orin Nano (8GB) / Jetson Xavier NX, or a dedicated local workstation with an NVIDIA GTX/RTX GPU (minimum 4GB VRAM) for real-time YOLOv8 inference at 30+ FPS.

CPU Fallback Option: Intel Core i5/i7 (11th Gen or newer) or AMD Ryzen 5/7 with at least 16GB RAM (suitable for frame-skipped processing).

Storage: 256GB+ NVMe SSD for storing local SQLite databases (video_data.db), model weights, and rolling video buffers.

2. Optical & Camera Hardware

IP CCTV Cameras: 1080p (Full HD) or 4K PoE (Power over Ethernet) wide-angle dome/bullet cameras supporting RTSP (Real-Time Streaming Protocol) streams.

Lens Specifications: 2.8mm wide-angle lens (90°–110° Field of View) mounted overhead at a 30°–45° downward tilt to minimize occlusion and accurately capture foot coordinates within defined zones.

Low-Light / IR Capability: Built-in Infrared (IR) night vision and WDR (Wide Dynamic Range) to handle varying store lighting conditions near glass entrances.

3. Networking & Power Infrastructure

PoE Network Switch: 8-Port Gigabit PoE+ Switch to power multiple IP cameras and transmit high-bitrate video streams over single Cat6 Ethernet cables.

UPS (Uninterruptible Power Supply): 1000VA / 600W Line-Interactive UPS to prevent database corruption and ensure continuous tracking during brief power fluctuations.

4. Optional In-Store Engagement Peripherals

Digital Signage Displays: Commercial IPS screens placed in the Main Display Shelf Zone or Entrance Corridor to trigger dynamic promotions when dwell time or footfall thresholds are met.

Software Tech Stack

Computer Vision: Python, OpenCV, Ultralytics YOLOv8, deep-sort-realtime

Machine Learning & Optimization: Scikit-learn, SciPy, NumPy, Pandas, Joblib

Backend & Database: FastAPI, Uvicorn, SQLite, SQLAlchemy

Frontend & Visualization: Streamlit, Plotly

Generative AI: Google Gemini API (google-genai)

Quick Start & Installation

1. Clone the Repository

git clone https://github.com/suryanshvaish1/autonoumous-retail-intelligence-system.git
cd autonoumous-retail-intelligence-system


2. Set Up Virtual Environment & Install Dependencies

python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt


3. Calibrate Store Zones (First-Time Setup)

Define custom bounding boxes over your store video feed so the tracker accurately logs zone dwell times:

python vision_pipeline.py --video store.mp4 --setup-zones --fresh


(Drag rectangles with your mouse over each zone, press ENTER to confirm each box, and press q when finished.)

4. Run the Vision Pipeline

Process a store video feed and populate video_data.db:

python vision_pipeline.py --video store.mp4 --fresh


5. Launch the Backend & Dashboard

Open two separate terminals (with venv activated):

Terminal 1 — FastAPI Backend:

uvicorn api:app --reload


Terminal 2 — Streamlit Dashboard:

streamlit run dashboard.py


Navigate to http://localhost:8501 in your browser to explore the live analytics dashboard.

Project Structure

├── api.py                  # FastAPI REST endpoints for telemetry & analytics
├── dashboard.py            # Main 8-tab Streamlit executive dashboard
├── vision_pipeline.py      # YOLOv8 + DeepSORT detection, tracking & heatmap generator
├── generator.py            # Synthetic data generator & behavioral simulator
├── zones.json              # Saved spatial coordinates for monitored store zones
├── requirements.txt        # Python package dependencies
├── .gitignore              # Excludes venv, heavy model weights, and local DBs
└── README.md               # Project documentation
