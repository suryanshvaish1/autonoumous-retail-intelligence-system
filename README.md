# 🛒 Autonomous Retail Intelligence System

An end-to-end computer vision and predictive analytics platform designed for retail stores. The system tracks customer footfall, dwell time, and spatial movement in real time using **YOLOv8** and **DeepSORT**, logs behavioral metrics into SQLite databases, and translates raw video feeds into actionable business insights via an interactive **Streamlit** and **FastAPI** executive dashboard.

---

## 🏗️ System Architecture & Block Diagram

The system operates across four sequential stages—from raw optical video ingestion to executive business intelligence:

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
