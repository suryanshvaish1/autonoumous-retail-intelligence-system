# 🛒 Autonomous Retail Intelligence System

An end-to-end computer vision and predictive analytics platform designed for retail stores. The system tracks customer footfall, dwell time, and spatial movement in real time using **YOLOv8** and **DeepSORT**, logs behavioral metrics into SQLite databases, and translates raw video feeds into actionable business insights via an interactive **Streamlit** and **FastAPI** executive dashboard.

---

## 🏗️ System Architecture & Block Diagram

The system operates across four sequential stages—from raw optical video ingestion to executive business intelligence.

```text
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
```

### Block Diagram Workflow Explanation

1. **Video Ingestion:** Raw footage is captured either via live RTSP IP CCTV camera feeds or uploaded MP4 test files.

2. **Computer Vision & Tracking Engine (`vision_pipeline.py`):**
   - Every video frame passes through **YOLOv8n** to detect human bounding boxes.
   - **DeepSORT** assigns persistent tracking IDs across consecutive frames so individual shoppers are not double-counted.
   - Foot coordinates are checked against polygon boundaries defined in `zones.json` to calculate exact dwell times.
   - Accumulated coordinates generate `heatmap_output.png`.

3. **Data Storage & Simulation (`video_data.db` & `generator.py`):**
   - Live visitor sessions exceeding the minimum dwell threshold are committed to SQLite (`video_data.db`).
   - For multi-week forecasting, back-testing, and model training, `generator.py` supplies realistic 75-day retail behavioral datasets (`retail_data.db`).

4. **API & Executive Dashboard (`api.py` & `dashboard.py`):**
   - `api.py` exposes structured REST endpoints for external integrations and telemetry queries.
   - `dashboard.py` renders the 8-tab Streamlit command center, running Gradient Boosting purchase classifiers, multiple-choice knapsack budget optimization, time-series staffing forecasts, K-Means customer segmentation, and Google Gemini AI strategic recommendations.

---

## ✨ Key Features

- **Real-Time Multi-Object Tracking:** Detects and tracks unique shoppers across customizable store zones (`Entrance Corridor`, `Main Display Shelf Zone`, `Checkout Counter`) using YOLOv8 and DeepSORT.
- **Spatial Movement Heatmaps:** Automatically generates Gaussian movement heatmaps (`heatmap_output.png`) highlighting high-traffic corridors and engagement hotspots.
- **Predictive Purchase Modeling:** Evaluates visitor purchase probability using a Gradient Boosting classifier trained on dwell time, zone activity, and demographic features.
- **Revenue & Budget Optimization:** Uses dynamic programming (multiple-choice knapsack solver) to recommend the highest-ROI marketing and store-layout interventions within a fixed budget.
- **7-Day Footfall Forecasting & Staffing:** Back-tests time-series forecasting models (Seasonal Average vs. Ridge Regression) to predict upcoming store traffic and calculate optimal hourly staff schedules.
- **A/B Test Planner & Customer Segmentation:** Uses K-Means clustering to segment shoppers by intent and calculates statistically valid sample sizes for in-store promotional experiments.
- **AI Retail Advisor:** Integrates Google Gemini to provide natural-language executive summaries, multi-period comparisons, and interactive Q&A.

---

## 📁 Codebase & Module Breakdown

Below is a detailed breakdown of what each core Python module in the repository analyzes and executes.

### 1. `vision_pipeline.py` — Computer Vision & Spatial Telemetry

- Processes store CCTV video feeds frame by frame using YOLOv8 for person detection and DeepSORT for multi-object tracking.
- Evaluates shopper foot positions against custom spatial boundaries (`zones.json`).
- Measures exact dwell times per zone.
- Renders Gaussian movement heatmaps.
- Writes verified visitor session logs into `video_data.db`.

### 2. `dashboard.py` — Executive Analytics & ML Interface

- Powers the interactive 8-tab Streamlit user interface.
- Ingests either live camera logs (`video_data.db`) or 75-day historical datasets.
- Analyzes traffic patterns, demographic distributions, and zone dwell metrics.
- Executes Gradient Boosting purchase probability predictions.
- Runs dynamic programming (knapsack) budget optimizations.
- Performs K-Means customer clustering.
- Evaluates model ROC-AUC/calibration metrics.
- Connects with Google Gemini for AI-driven store recommendations.

### 3. `api.py` — Backend Telemetry & REST Service

- Manages the FastAPI backend server and RESTful endpoints.
- Queries and aggregates store telemetry, zone performance metrics, visitor logs, and forecasting outputs from the underlying SQLite databases.
- Serves clean JSON responses for external clients and microservices.

### 4. `generator.py` — Synthetic Retail Data & Behavioral Simulator

- Models and generates realistic multi-week retail datasets covering 75 days of:
  - Customer footfall
  - Age/gender demographics
  - Zone dwell times
  - Promotional lift
  - Purchase outcomes
- Provides a statistically rich baseline for training machine learning classifiers, running 14-day forecasting back-tests, and demonstrating full-scale dashboard analytics when only short CCTV clips are available.

---

## 💻 Hardware Architecture (Edge & Deployment Setup)

While the system can run locally on a standard laptop using pre-recorded video files, deploying it in a live physical retail environment requires the following hardware components.

### 1. Edge Compute & Processing Unit

- **Primary Edge Node:** NVIDIA Jetson Orin Nano (8GB) / Jetson Xavier NX, or a dedicated local workstation with an NVIDIA GTX/RTX GPU (minimum 4GB VRAM) for real-time YOLOv8 inference at 30+ FPS.
- **CPU Fallback Option:** Intel Core i5/i7 (11th Gen or newer) or AMD Ryzen 5/7 with at least 16GB RAM (suitable for frame-skipped processing).
- **Storage:** 256GB+ NVMe SSD for storing local SQLite databases (`video_data.db`), model weights, and rolling video buffers.

### 2. Optical & Camera Hardware

- **IP CCTV Cameras:** 1080p (Full HD) or 4K PoE (Power over Ethernet) wide-angle dome/bullet cameras supporting RTSP (Real-Time Streaming Protocol) streams.
- **Lens Specifications:** 2.8mm wide-angle lens (90°–110° Field of View) mounted overhead at a 30°–45° downward tilt to minimize occlusion and accurately capture foot coordinates within defined zones.
- **Low-Light / IR Capability:** Built-in Infrared (IR) night vision and WDR (Wide Dynamic Range) to handle varying store lighting conditions near glass entrances.

### 3. Networking & Power Infrastructure

- **PoE Network Switch:** 8-Port Gigabit PoE+ Switch to power multiple IP cameras and transmit high-bitrate video streams over single Cat6 Ethernet cables.
- **UPS (Uninterruptible Power Supply):** 1000VA / 600W Line-Interactive UPS to prevent database corruption and ensure continuous tracking during brief power fluctuations.

### 4. Optional In-Store Engagement Peripherals

- **Digital Signage Displays:** Commercial IPS screens placed in the *Main Display Shelf Zone* or *Entrance Corridor* to trigger dynamic promotions when dwell time or footfall thresholds are met.

---

## 🛠️ Software Tech Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Computer Vision | OpenCV, Ultralytics YOLOv8, DeepSORT |
| Machine Learning & Optimization | Scikit-learn, SciPy, NumPy, Pandas, Joblib |
| Backend & Database | FastAPI, Uvicorn, SQLite, SQLAlchemy |
| Frontend & Visualization | Streamlit, Plotly |
| Generative AI | Google Gemini API (`google-genai`) |

---

## 🚀 Quick Start & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/suryanshvaish1/autonoumous-retail-intelligence-system.git

cd autonoumous-retail-intelligence-system
```

### 2. Set Up Virtual Environment & Install Dependencies

**Windows PowerShell:**

```powershell
python -m venv venv

.\venv\Scripts\Activate.ps1

python -m pip install --upgrade pip

pip install -r requirements.txt
```

**Linux / macOS:**

```bash
python3 -m venv venv

source venv/bin/activate

python -m pip install --upgrade pip

pip install -r requirements.txt
```

### 3. Calibrate Store Zones (First-Time Setup)

Define custom bounding boxes over your store video feed so the tracker accurately logs zone dwell times.

```bash
python vision_pipeline.py --video store.mp4 --setup-zones --fresh
```

**Instructions:**

1. The video window will open.
2. Drag rectangles with your mouse over each store zone.
3. Press **ENTER** to confirm each zone.
4. Press `q` when finished.

The configured zone coordinates are saved in `zones.json`.

### 4. Run the Vision Pipeline

Process a store video feed and populate `video_data.db`.

```bash
python vision_pipeline.py --video store.mp4 --fresh
```

The pipeline:

1. Reads the input video.
2. Detects people using YOLOv8.
3. Tracks visitors using DeepSORT.
4. Identifies their positions inside configured store zones.
5. Calculates zone-specific dwell times.
6. Generates movement heatmaps.
7. Stores visitor session data in SQLite.

Generated output:

```text
video_data.db
heatmap_output.png
zones.json
```

### 5. Launch the Backend & Dashboard

Open two separate terminals with the virtual environment activated.

**Terminal 1 — FastAPI Backend:**

```bash
uvicorn api:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

**Terminal 2 — Streamlit Dashboard:**

```bash
streamlit run dashboard.py
```

Navigate to:

```text
http://localhost:8501
```

to explore the retail analytics dashboard.

---

## 📊 Analytics & Machine Learning

The platform combines real-time computer vision telemetry with historical and synthetic retail datasets.

### Predictive Purchase Modeling

Uses a Gradient Boosting classifier to estimate purchase probability based on visitor behavior, including:

- Zone activity
- Dwell time
- Visitor demographics
- Historical purchase outcomes

### Revenue & Budget Optimization

Uses a multiple-choice knapsack optimization approach to recommend marketing and store-layout interventions within a defined budget.

### Footfall Forecasting

Uses time-series forecasting approaches, including:

- Seasonal Average
- Ridge Regression

The models help estimate upcoming visitor traffic and support staffing decisions.

### Customer Segmentation

Uses K-Means clustering to group shoppers based on behavioral patterns and engagement.

### A/B Testing

Provides an A/B test planning workflow to calculate statistically valid sample sizes for in-store promotional experiments.

### AI Retail Advisor

Integrates Google Gemini to provide:

- Natural-language executive summaries
- Multi-period performance comparisons
- Interactive business analytics Q&A
- Retail operational recommendations

---

## 📡 REST API

The FastAPI backend exposes structured analytics for external applications.

Start the backend:

```bash
uvicorn api:app --reload
```

Open the interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

The API provides access to visitor telemetry, zone performance, and analytics through REST endpoints.

---

## 🔄 End-to-End Data Flow

```text
                 CCTV / Video
                      |
                      v
              OpenCV Frame Reader
                      |
                      v
                YOLOv8 Detection
                      |
                      v
              DeepSORT Tracking
                      |
             +--------+--------+
             |                 |
             v                 v
       Visitor Analytics   Movement Data
             |                 |
       +-----+-----+           v
       |     |     |      Heatmap Engine
       v     v     v           |
      Zone  Dwell Demographic  v
       |     |     |    heatmap_output.png
       +-----+-----+
             |
             v
          SQLite DB
             |
       +-----+------+
       |            |
       v            v
    FastAPI     ML Analytics
       |            |
       +------+-----+
              |
              v
      Streamlit Dashboard
              |
              v
       Executive Insights
              |
              v
       Gemini AI Advisor
```

---

## 🎯 Business Use Cases

### Store Layout Optimization

Identify areas receiving high or low customer traffic and use those patterns to inform product placement and store-layout decisions.

### Customer Engagement

Measure how long visitors spend in specific product areas.

### Checkout Monitoring

Identify long dwell times around checkout areas and detect potential congestion.

### Staffing Optimization

Use traffic patterns to understand when additional employees may be required.

### Product Placement Analysis

Compare visitor activity around different store zones.

### Customer Journey Analysis

Analyze visitor movement between different store areas using tracked visitor trajectories.

### Executive Reporting

Generate summarized analytics and reports for store managers and business stakeholders.

---

## 🔐 Privacy & Responsible AI

This project is intended as a technical retail-analytics prototype.

For production deployments, additional considerations should be implemented:

- Appropriate customer consent and notification.
- Compliance with applicable privacy regulations.
- Secure data storage and transmission.
- Data-retention and deletion policies.
- Role-based access control.
- Model-accuracy evaluation.
- Bias and fairness evaluation.
- Appropriate handling of CCTV footage.
- Clear policies for access to and use of analytics.

Where possible, production deployments should minimize the retention of identifiable video and use aggregated analytics for reporting.

---

## 📈 Future Improvements

Potential improvements include:

- 🔴 Live multi-camera CCTV/IP streaming.
- 🧠 More advanced customer-behavior analysis.
- 📦 Product detection and shelf monitoring.
- 🛍️ Product-level interaction tracking.
- 🚨 Real-time anomaly detection.
- 📱 Mobile analytics dashboard.
- ☁️ Cloud-based deployment.
- 🔔 Real-time alerts through Slack/email.
- 📊 Historical trend analysis.
- 👥 Customer-journey visualization.
- ⚡ TensorRT / ONNX edge-GPU optimization.
- 🔄 Real-time streaming analytics.
- 🏬 Multi-store analytics.
- 📈 Advanced retail forecasting models.

---

## 📂 Project Structure

```text
autonomous-retail-intelligence-system/
│
├── api.py                  # FastAPI REST endpoints for telemetry and analytics
├── dashboard.py            # Main 8-tab Streamlit executive dashboard
├── vision_pipeline.py      # YOLOv8 + DeepSORT detection, tracking and heatmap generator
├── generator.py            # Synthetic data generator and behavioral simulator
├── zones.json              # Saved spatial coordinates for monitored store zones
├── requirements.txt        # Python package dependencies
├── .gitignore              # Excludes virtual environments, model weights and local databases
└── README.md               # Project documentation
```

---

## 📜 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Suryansh Vaish**

- GitHub: [suryanshvaish1](https://github.com/suryanshvaish1)
- Repository: [Autonomous Retail Intelligence System](https://github.com/suryanshvaish1/autonoumous-retail-intelligence-system)

---

## ⭐ If You Find This Project Useful

Consider giving the repository a ⭐ on GitHub!
