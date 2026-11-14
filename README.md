Autonomous Retail Intelligence System

An end-to-end computer vision and predictive analytics platform designed for retail stores. The system tracks customer footfall, dwell time, and spatial movement in real time using YOLOv8 and DeepSORT, logs behavioral metrics into SQLite databases, and translates raw video feeds into actionable business insights via an interactive Streamlit and FastAPI executive dashboard.

Key Features

Real-Time Multi-Object Tracking: Detects and tracks unique shoppers across customizable store zones (Entrance Corridor, Main Display Shelf Zone, Checkout Counter) using YOLOv8 and DeepSORT.

Spatial Movement Heatmaps: Automatically generates Gaussian movement heatmaps (heatmap_output.png) highlighting high-traffic corridors and engagement hotspots.

Predictive Purchase Modeling: Evaluates visitor purchase probability using a Gradient Boosting classifier trained on dwell time, zone activity, and demographic features.

Revenue & Budget Optimization: Uses dynamic programming (multiple-choice knapsack solver) to recommend the highest-ROI marketing and store-layout interventions within a fixed budget.

7-Day Footfall Forecasting & Staffing: Back-tests time-series forecasting models (Seasonal Average vs. Ridge Regression) to predict upcoming store traffic and calculate optimal hourly staff schedules.

A/B Test Planner & Customer Segmentation: Uses K-Means clustering to segment shoppers by intent and calculates statistically valid sample sizes for in-store promotional experiments.

AI Retail Advisor: Integrates Google Gemini to provide natural-language executive summaries, multi-period comparisons, and interactive Q&A.

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
├── train_model.py          # Synthetic data generator & Gradient Boosting model trainer
├── zones.json              # Saved spatial coordinates for monitored store zones
├── requirements.txt        # Python package dependencies
├── .gitignore              # Excludes venv, heavy model weights, and local DBs
└── README.md               # Project documentation
