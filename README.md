# Autonomous Retail Intelligence System

**Real-Time Store Analytics & Customer Footfall Insights**

VTU BEC685 Major Project (Phase 1), Electronics & Communication Engineering

An end-to-end system that turns store camera video into business insights. It detects and tracks customers, measures where they spend time, saves the results to a database, and shows predictions, forecasts, budgeting guidance and AI advice on an interactive dashboard.

```
Video / Webcam -> YOLOv8 detection -> DeepSORT tracking -> Zones + Dwell time -> Heatmap
              -> SQLite -> FastAPI -> Streamlit Dashboard (ML, forecasting, Gemini AI Advisor)
```

---

## Quick Start: All Running Commands (Copy and Paste)

Run these in **Windows PowerShell** from the project folder. Activate the virtual environment in **every** new terminal.

### 1. One-time setup

```powershell
git clone https://github.com/YOUR-USERNAME/retail-intelligence-system.git
cd retail-intelligence-system

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
pip install ultralytics deep-sort-realtime opencv-python
pip install "setuptools<81"

python generate_mock_db.py
python train_model.py
```

If PowerShell blocks activation, run this once and try again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

Check the install:

```powershell
python -c "import ultralytics, deep_sort_realtime, cv2; print('all good')"
```

### 2. Every time you run the project

**Terminal 1: API**

```powershell
cd "D:\retail intelligence\retail-intelligence-system"
venv\Scripts\activate
uvicorn api:app --reload
```

**Terminal 2: Dashboard** (open a new terminal)

```powershell
cd "D:\retail intelligence\retail-intelligence-system"
venv\Scripts\activate
streamlit run dashboard.py
```

Then open http://localhost:8501. In the sidebar, **Data Source** switches between **Mock data (75 days)** and **Live video pipeline**.

### 3. Process a video

**Option A: upload in the dashboard.** Choose **Live video pipeline**, upload a video in the sidebar, then click **Process video**.

**Option B: Terminal 3 (with live preview window)**

```powershell
cd "D:\retail intelligence\retail-intelligence-system"
venv\Scripts\activate

# First time only: draw the 3 zones with the mouse (ENTER = confirm, C = skip)
python vision_pipeline.py --video store.mp4 --setup-zones --fresh

# Next runs (reuses zones.json)
python vision_pipeline.py --video store.mp4 --fresh

# Slow laptop
python vision_pipeline.py --video store.mp4 --fresh --skip 3 --width 640

# Detect more people
python vision_pipeline.py --video store.mp4 --fresh --conf 0.3

# Save the annotated video
python vision_pipeline.py --video store.mp4 --fresh --save-video output.mp4

# Laptop camera
python vision_pipeline.py --webcam --fresh
```

Press **q** in the preview window to stop early (data is still saved). Then refresh the dashboard (F5).

### 4. Reset or rebuild

```powershell
python generate_mock_db.py                          # rebuild mock data only
python vision_pipeline.py --video store.mp4 --fresh # clear and rebuild video data only
python train_model.py                               # retrain the model
```

---

## Project Explained Using the STAR Method

### S - Situation

Small and mid-size retail stores usually do not know how customers actually move inside the store. Managers count sales at the billing counter, but they cannot see:

- which areas of the store attract attention and which are ignored,
- how long customers stay in front of a display,
- which visitors are likely to buy, and how many staff or how much promotion budget is needed on a given day.

Cameras are already installed in most stores, but the footage is only used for security, not for decisions. Commercial footfall-analytics products exist, but they are expensive and closed. Also, in this project no long real store video was available, so the system also needed a way to demonstrate what insights look like after weeks of data.

### T - Task

Build a complete, low-cost and explainable analytics system that:

1. Converts raw store video into structured visitor data (detection, tracking, zones, dwell time, heatmap).
2. Stores the data in a database and exposes it through an API.
3. Gives managers an easy dashboard with predictions, revenue and budget guidance, forecasts, staffing advice and alerts.
4. Provides an AI assistant that explains the numbers in plain language.
5. Works both on real video and on a 75-day mock dataset, so the long-term insights can be demonstrated, and lets the user switch between the two.

### A - Action

| Step | What was done |
|---|---|
| **1. Vision pipeline** (`vision_pipeline.py`) | YOLOv8n detects people in every Nth frame. DeepSORT gives each person a stable ID. The position of the person's feet decides which zone they are in (Entrance Corridor, Main Display Shelf Zone, Checkout Counter). Dwell time is accumulated per zone. A person is logged as one visitor when they leave the scene. |
| **2. Heatmap** | Every footprint is added to an accumulator, blurred with a Gaussian filter, colour-mapped and blended over the first video frame. Saved as `heatmap_output.png`. |
| **3. Two data sources** (`data_sources.py`) | Two SQLite databases with the same table layout: `retail_data.db` (75 days of mock data, about 4,400 rows) and `video_data.db` (rows from a real video). They never mix. |
| **4. Backend API** (`api.py`) | FastAPI endpoints return summaries, predictions and analytics. Every endpoint accepts `?source=mock` or `?source=video`. |
| **5. Machine learning** (`train_model.py`, `insights.py`) | A model trained on a synthetic visitor dataset predicts purchase likelihood from zone, dwell time and visitor attributes. It powers the "What-If" simulator. |
| **6. Analytics modules** (`analytics.py`, `advanced.py`, `period_compare.py`) | Revenue and budget estimates, customer segments, demand forecasting, staffing suggestions, alerts and period comparison. |
| **7. AI Advisor** (`gemini_advisor.py`) | Google Gemini answers manager questions (for example about budgeting) using the store's own numbers. |
| **8. Dashboard** (`dashboard.py`) | Streamlit dashboard with 8 tabs. A sidebar switch changes between mock data and live video data. A video can be uploaded directly in the sidebar and is processed in the background. |

**Tabs:** Overview, Predictions & What-If, Revenue & Budget, Forecast & Staffing, Customer Segments, Alerts, Model Lab, AI Advisor.

### R - Result

- A working end-to-end system: video in, decisions out, in one dashboard.
- Managers can see **where** customers go (heatmap), **how long** they stay (dwell time) and **who** is likely to buy (prediction), then test ideas with the What-If simulator.
- The mock-data mode shows what 2 months of insights look like (forecast, alerts, revenue, staffing), while the video mode shows how raw footage becomes data.
- Model performance on the synthetic dataset: **accuracy 0.666, ROC-AUC 0.704**. These numbers describe the demo dataset, not a real store.
- The design is modular: each part (vision, database, API, ML, dashboard, AI) can be improved or replaced independently.

---

## Honest Limitations

- The ML model is trained on **synthetic data**. Its numbers are for demonstration, and the What-If tool is not causal proof.
- **Gender and age in the video pipeline are placeholders** assigned from the track ID. No face analysis is done, on purpose, for privacy. They are not real demographic recognition.
- One row is saved per visitor, using the zone where they stayed longest.
- A short video makes time-based tabs (trends, alerts, forecast) thin or inactive. Alerts need 14+ days of data and forecasting needs 28+ days. This is why the mock data source exists.
- Multi-camera re-identification, action recognition and real age/gender estimation are out of scope for this phase.

---

## Project Structure

| File | Purpose |
|---|---|
| `vision_pipeline.py` | Detection, tracking, zones, dwell time, heatmap, writes `video_data.db` |
| `data_sources.py` | Maps source name (`mock` / `video`) to its database |
| `database.py` | Table definition (`VisitorEvent`) |
| `generate_mock_db.py` | Creates the 75-day mock database |
| `train_model.py` | Trains the purchase-prediction model (`model.joblib`) |
| `insights.py` | Predictions and What-If logic |
| `analytics.py`, `advanced.py`, `period_compare.py` | Analytics, forecasting, staffing, alerts, period comparison |
| `gemini_advisor.py` | Gemini AI Advisor |
| `api.py` | FastAPI backend |
| `dashboard.py` | Streamlit dashboard |
| `synthetic_retail_visitor_data.csv` | Synthetic dataset used for training |

---

## Setup

Requires **Python 3.12** and Git.

```bash
git clone https://github.com/YOUR-USERNAME/retail-intelligence-system.git
cd retail-intelligence-system

python -m venv venv
venv\Scripts\activate          # Windows PowerShell

pip install -r requirements.txt
pip install "setuptools<81"    # needed by deep-sort-realtime
```

**Gemini API key:** the AI Advisor needs your own Google Gemini API key. Set it as described in `gemini_advisor.py` (use an environment variable or a `.env` file, and never commit the key).

The YOLO model file `yolov8n.pt` (about 6 MB) downloads automatically on the first run.

### Create the data

```bash
python generate_mock_db.py     # creates retail_data.db (mock data)
python train_model.py          # creates model.joblib (if not already present)
```

---

## How to Run

Activate the virtual environment in **every** terminal.

**Terminal 1: API**
```bash
uvicorn api:app --reload
```

**Terminal 2: Dashboard**
```bash
streamlit run dashboard.py
```

Open the address shown by Streamlit (usually http://localhost:8501).

### Using video

**Option A: upload in the dashboard.** In the sidebar choose **Live video pipeline**, upload a video, and click **Process video**. The dashboard fills in when processing finishes.

**Option B: command line (with live preview window).**
```bash
# First time: draw the 3 zones with the mouse (ENTER to confirm, C to skip)
python vision_pipeline.py --video store.mp4 --setup-zones --fresh

# Later runs (reuses zones.json)
python vision_pipeline.py --video store.mp4 --fresh

# Laptop camera
python vision_pipeline.py --webcam --fresh
```

| Flag | Meaning |
|---|---|
| `--video FILE` | Video file to process |
| `--webcam` | Use the laptop camera |
| `--fresh` | Clear old video data first (mock data is never touched) |
| `--setup-zones` | Draw zones with the mouse and save `zones.json` |
| `--no-show` | Do not open the preview window |
| `--save-video out.mp4` | Save the annotated video |
| `--skip N` | Analyse every Nth frame (use 3 for slow laptops) |
| `--conf 0.4` | Detection confidence (lower, e.g. 0.3, finds more people) |
| `--width 960` | Resize frames to this width |

### Switching data source

Use **Data Source** in the dashboard sidebar:

- **Mock data (75 days):** shows what long-term insights look like.
- **Live video pipeline:** shows data produced from your video, plus the heatmap.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `No module named 'pkg_resources'` | `pip install "setuptools<81"` |
| `ModuleNotFoundError` | Activate `venv` in that terminal and run `pip install -r requirements.txt` |
| Dashboard says it cannot connect to FastAPI | Start `uvicorn api:app --reload` |
| "The video database is empty" | Process a video first, or lower `--conf` and check that zones cover where people walk |
| Very few visitors found | Redo `--setup-zones`, or delete `zones.json` to return to the default zones |
| Processing is slow | Use `--skip 3` and/or `--width 640` |
| No preview window / GUI error | Install `opencv-python` (not `opencv-python-headless`), or use `--no-show` |

---

## Tech Stack

Python, YOLOv8 (Ultralytics), DeepSORT, OpenCV, SQLite, SQLAlchemy, FastAPI, Uvicorn, Streamlit, Plotly, scikit-learn, Google Gemini API.

---

## Author

Your Name, your college and your USN here.
