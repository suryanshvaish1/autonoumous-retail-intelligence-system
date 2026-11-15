# 🛒 Autonomous Retail Intelligence System
### AI-Powered Smart Retail Store Monitoring and Automation Using Computer Vision, IoT and Embedded Systems

An intelligent **AI + IoT + Computer Vision platform** designed to transform traditional retail stores into smart, data-driven environments.

The system combines real-time customer detection, multi-object tracking, spatial analytics, machine learning, embedded electronics and IoT-based monitoring to convert CCTV footage and sensor data into actionable retail intelligence.

Using **YOLOv8, DeepSORT, ESP32, Python, FastAPI and Streamlit**, the platform monitors customer movement, calculates zone-specific dwell times, analyzes store traffic, generates foot-traffic heatmaps and provides intelligent operational recommendations.

The electronics extension integrates physical sensors, an LCD display, LEDs and a buzzer to create a complete hardware-software retail monitoring prototype.

---

# 📌 1. Project Overview

Traditional retail stores often lack intelligent systems to understand customer movement, identify crowded areas and analyze customer engagement.

The Autonomous Retail Intelligence System addresses these challenges by combining computer vision, machine learning and embedded electronics.

The system is designed around three major components:

### 1. Computer Vision & Artificial Intelligence

- Real-time person detection using YOLOv8.
- Multi-object tracking using DeepSORT.
- Customer movement tracking.
- Zone-specific dwell-time calculation.
- Foot-traffic heatmap generation.
- Estimated demographic classification.
- Predictive purchase analytics.

### 2. IoT & Embedded Electronics

- ESP32-based sensor integration.
- IR-based entry and exit detection.
- HC-SR04 ultrasonic distance measurement.
- PIR-based motion detection.
- Real-time visitor counter.
- LCD-based information display.
- LED-based occupancy indicators.
- Buzzer-based alert system.

### 3. Backend & Business Intelligence

- FastAPI REST backend.
- SQLite database.
- Streamlit executive dashboard.
- Machine learning analytics.
- AI-powered retail recommendations.
- Customer behavior analysis.
- Store performance monitoring.

---

# 🏗️ 2. Complete System Architecture

The system is organized into five major layers.

```text
                  SMART RETAIL STORE
                          |
             +------------+------------+
             |                         |
             v                         v
       CCTV / WEBCAM              ELECTRONIC SENSORS
             |                         |
             v                         v
       COMPUTER VISION                 ESP32
             |                         |
             v                         v
          YOLOv8                  SENSOR PROCESSING
             |                         |
             v                         |
          DeepSORT                      |
             |                         |
             v                         |
       CUSTOMER TRACKING                |
             |                         |
             +------------+------------+
                          |
                          v
                   PYTHON BACKEND
                          |
              +-----------+-----------+
              |                       |
              v                       v
         SQLite Database        Analytics Engine
              |                       |
              +-----------+-----------+
                          |
                          v
                      FastAPI
                          |
                          v
                  Streamlit Dashboard
                          |
              +-----------+-----------+
              |           |           |
              v           v           v
             LCD         LED        BUZZER
              |           |           |
              +-----------+-----------+
                          |
                          v
                 STORE MANAGER
```

## Architecture Workflow

### Layer 1: Video Ingestion

The system receives video from:

- CCTV/IP cameras.
- RTSP video streams.
- USB webcams.
- Pre-recorded MP4 videos.

OpenCV captures and processes video frames.

### Layer 2: Computer Vision & Tracking

The computer vision pipeline uses:

**YOLOv8:**
- Detects people in video frames.
- Identifies bounding boxes.
- Extracts detection confidence scores.

**DeepSORT:**
- Assigns unique tracking IDs.
- Maintains visitor identities across frames.
- Tracks movement between store zones.
- Helps prevent duplicate visitor counting.

### Layer 3: Embedded Electronics & IoT

The ESP32 communicates with connected electronic sensors.

It receives information from:

- IR sensors.
- Ultrasonic sensors.
- PIR motion sensors.

Based on the sensor readings, it can:

- Detect entrance and exit events.
- Monitor movement.
- Measure distance.
- Update visitor counts.
- Trigger occupancy alerts.

The ESP32 communicates with the Python backend through Wi-Fi or serial communication.
![ESP32 Hardware Circuit Diagram](assets/circuit-diagram.png)

### Layer 4: Data Processing & Storage

The backend combines:

- Computer vision telemetry.
- Sensor readings.
- Visitor movement data.
- Dwell-time information.
- Historical retail analytics.

SQLite stores the processed information.

### Layer 5: Dashboard & Automation

The Streamlit dashboard displays:

- Total visitors.
- Store occupancy.
- Zone-specific dwell times.
- Customer movement.
- Foot-traffic heatmaps.
- Predictive analytics.
- Operational recommendations.

The electronic outputs provide additional physical feedback through LCD, LEDs and buzzer alerts.

---

# ✨ 3. Key Features

## Computer Vision Features

- 🎯 Real-Time Person Detection using YOLOv8.
- 🧭 Multi-Object Tracking using DeepSORT.
- 📍 Zone / ROI Detection.
- ⏱️ Visitor Dwell-Time Analytics.
- 🗺️ 2D Foot-Traffic Heatmaps.
- 🧑 Estimated Demographic Classification.
- 👥 Customer Movement Analysis.

## Machine Learning Features

- 📈 Predictive Purchase Modeling.
- 📊 Customer Segmentation using K-Means.
- 🔮 Seven-Day Footfall Forecasting.
- 💰 Revenue and Budget Optimization.
- 🧪 A/B Testing Planner.
- 📉 Model Performance Evaluation.

## Electronics & IoT Features

- 🔌 ESP32 Microcontroller Integration.
- 🚶 IR-Based Entry and Exit Detection.
- 📡 HC-SR04 Distance Measurement.
- 🏃 PIR Motion Detection.
- 🔢 Automatic Visitor Counting.
- 🖥️ LCD-Based Store Monitoring.
- 🟢 Green LED Occupancy Indicator.
- 🔴 Red LED Alert Indicator.
- 🔔 Buzzer-Based Notifications.

## Backend & Dashboard Features

- ⚡ FastAPI REST Backend.
- 🗄️ SQLite Database.
- 📊 Interactive Streamlit Dashboard.
- 🤖 Gemini AI Retail Advisor.
- 📄 Retail Analytics Reporting.
- 📥 Data Export.
- 📡 Sensor Data Integration.

---

# 🎥 4. Computer Vision & AI Pipeline

## 4.1 Video Ingestion

The system supports different video input sources:

- Pre-recorded MP4 videos.
- USB webcams.
- RTSP/IP CCTV streams.

OpenCV's `VideoCapture` reads the video frames sequentially.

## 4.2 Person Detection — YOLOv8

YOLOv8 is responsible for detecting people in the video.

The processing pipeline:

1. Captures a video frame.
2. Sends the frame to YOLOv8.
3. Detects objects.
4. Filters detections belonging to the person class.
5. Extracts bounding boxes and confidence scores.

The detected people are then passed to the tracking module.

## 4.3 Multi-Object Tracking — DeepSORT

DeepSORT maintains the identity of detected people across consecutive frames.

For example:

```text
Frame 100 → Person detected → Track ID 17

Frame 101 → Same person detected → Track ID 17

Frame 102 → Same person detected → Track ID 17
```

This helps the system track individual visitors instead of counting every detection as a new visitor.

## 4.4 Zone / ROI Detection

The store is divided into predefined Regions of Interest (ROI).

Example:

```text
+---------------------------------------+
|                                       |
|             SHELF AREA                |
|                                       |
|        [ PRODUCT DISPLAY ]            |
|                                       |
|---------------------------------------|
|                                       |
|             CUSTOMER AREA             |
|                                       |
|---------------------------------------|
|                                       |
|              CHECKOUT                 |
|                                       |
+---------------------------------------+
```

The system checks the tracked person's representative floor position against the configured zone boundaries.

This determines which store area the visitor is occupying.

## 4.5 Dwell-Time Calculation

Dwell time measures how long a visitor stays inside a particular zone.

A basic frame-based calculation is:

```text
Dwell Time = Frames Spent in Zone / Video FPS

Example:

FPS = 30

Frames in Zone = 300

Dwell Time = 300 / 30

            = 10 seconds
```

The calculated dwell time is stored as visitor telemetry.

## 4.6 Demographic Classification

A pretrained classification model estimates:

- Gender.
- Age group.

These predictions can be associated with visitor tracking sessions.

**Note:** These are model estimates, not verified personal attributes.

## 4.7 Foot-Traffic Heatmap

The system collects tracked visitor coordinates and accumulates them into a 2D NumPy grid.

Processing:

```text
Visitor Coordinates
        |
        v
2D NumPy Grid
        |
        v
Gaussian Smoothing
        |
        v
Intensity Normalization
        |
        v
OpenCV JET Colormap
        |
        v
heatmap_output.png
```

The heatmap helps identify:

- High-traffic areas.
- Low-traffic areas.
- Customer hotspots.
- Congested zones.
- Product browsing patterns.
- Checkout congestion.

---

# 🔌 5. Electronics & IoT Hardware Integration

The electronics extension converts the software-based retail intelligence platform into a physical smart retail monitoring prototype.

The ESP32 acts as the primary microcontroller for sensor communication and electronic output control.

## 5.1 Hardware Components

| Component | Quantity | Purpose |
|---|---:|---|
| ESP32 Development Board | 1 | Main microcontroller |
| HC-SR04 Ultrasonic Sensor | 1 | Distance measurement |
| IR Sensor Module | 2 | Entry and exit detection |
| PIR Motion Sensor | 1 | Human movement detection |
| 16x2 LCD with I2C | 1 | Displays visitor count and occupancy |
| Buzzer | 1 | Alert notifications |
| Red LED | 1 | High occupancy indication |
| Green LED | 1 | Normal occupancy indication |
| Breadboard | 1 | Circuit assembly |
| Jumper Wires | 1 set | Component connections |
| Resistors | As required | Circuit protection |
| USB Power Supply | 1 | Powers the ESP32 and prototype |

### Estimated Hardware Budget

| Component | Approximate Cost |
|---|---:|
| ESP32 Development Board | ₹400 |
| HC-SR04 Ultrasonic Sensor | ₹100 |
| 2 IR Sensors | ₹150 |
| PIR Sensor | ₹120 |
| 16x2 LCD with I2C | ₹200 |
| Buzzer and LEDs | ₹100 |
| Breadboard and Jumper Wires | ₹200 |
| Resistors and Accessories | ₹100 |
| **Estimated Total** | **₹1,370** |

Prices are approximate and may vary depending on the seller.

The budget excludes the laptop and webcam. Existing devices can be reused.

## 5.2 ESP32 Microcontroller

The ESP32 is the main controller for the electronic components.

Responsibilities:

- Reads sensor values.
- Detects entry and exit events.
- Measures distance using the ultrasonic sensor.
- Receives motion detection signals.
- Updates visitor count.
- Controls LEDs and buzzer.
- Communicates with the Python backend.

The ESP32 supports Wi-Fi, which allows communication with the backend without requiring a separate networking module.

## 5.3 IR Sensors — Entry and Exit Detection

Two IR sensors can be positioned near the store entrance.

Example:

```text
       STORE ENTRANCE

       CUSTOMER ENTERING
               |
               v
          +---------+
          | IR 1    |
          +---------+
               |
               v
          +---------+
          | IR 2    |
          +---------+
               |
               v
          STORE INTERIOR
```

The order in which the sensors detect a person can help determine the direction of movement.

Example:

```text
IR 1 → IR 2 = Entry

IR 2 → IR 1 = Exit
```

The ESP32 updates the visitor count accordingly.

This requires suitable sensor positioning and direction-detection logic.

## 5.4 HC-SR04 Ultrasonic Sensor

The HC-SR04 measures the distance between the sensor and an object.

It works by:

1. Sending an ultrasonic pulse.
2. Waiting for the reflected signal.
3. Measuring the time taken by the signal.
4. Calculating the distance.

Distance formula:

```text
Distance = (Time × Speed of Sound) / 2
```

Example applications:

- Detecting objects near a checkout.
- Measuring distance in a designated area.
- Monitoring whether an area is occupied.

The sensor does not identify people or track individual customers.

## 5.5 PIR Motion Sensor

The PIR sensor detects changes in infrared radiation caused by movement.

It can be used for:

- Detecting movement near store zones.
- Activating monitoring when movement occurs.
- Providing additional occupancy signals.

Unlike YOLOv8, a PIR sensor cannot identify individual people.

## 5.6 LCD Display

A 16x2 LCD can display information received from the ESP32.

Example:

```text
+------------------------+
| SMART RETAIL SYSTEM    |
|                        |
| Visitors:  12          |
| Status: NORMAL         |
+------------------------+
```

Possible displayed information:

- Current visitor count.
- Occupancy status.
- Distance readings.
- Alert messages.

## 5.7 LED Indicators

Two LEDs provide visual status information.

**Green LED:**

Indicates normal store occupancy.

**Red LED:**

Indicates that occupancy has crossed a configured threshold.

Example:

```text
Occupancy < Threshold

Green LED = ON
Red LED   = OFF
Buzzer    = OFF
```

```text
Occupancy >= Threshold

Green LED = OFF
Red LED   = ON
Buzzer    = ON
```

## 5.8 Buzzer Alerts

The buzzer generates an audible notification when configured conditions are met.

Possible alert conditions:

- High store occupancy.
- Checkout congestion.
- Configured sensor threshold exceeded.

The actual alert logic is implemented through the ESP32 firmware and backend integration.

---

# 🔗 6. Electronics Circuit & Pin Connections

The following is a suggested ESP32 wiring configuration.

| Component | ESP32 Pin |
|---|---|
| IR Sensor 1 OUT | GPIO 18 |
| IR Sensor 2 OUT | GPIO 19 |
| HC-SR04 TRIG | GPIO 5 |
| HC-SR04 ECHO | GPIO 17 |
| PIR Sensor OUT | GPIO 23 |
| Buzzer | GPIO 25 |
| Green LED | GPIO 26 |
| Red LED | GPIO 27 |
| LCD SDA | GPIO 21 |
| LCD SCL | GPIO 22 |

### Important Electrical Safety Notes

- ESP32 GPIO operates at 3.3V logic.
- HC-SR04 ECHO generally outputs 5V and should be connected through a suitable voltage divider or level shifter.
- Check the output voltage of IR modules before connecting them to ESP32 GPIO.
- Use appropriate current-limiting resistors for LEDs.
- Do not power high-current loads directly from ESP32 GPIO pins.
- Ensure all connected modules share a suitable common ground.

The wiring configuration may need adjustment depending on the exact ESP32 board and sensor modules.

---

# 📡 7. ESP32 and Python Backend Communication

The ESP32 communicates with the Python backend using Wi-Fi.

Suggested communication flow:

```text
IR / PIR / ULTRASONIC SENSORS
              |
              v
             ESP32
              |
              |
           Wi-Fi
              |
              v
        Python Backend
              |
              v
          FastAPI
              |
              v
         SQLite DB
              |
              v
       Streamlit Dashboard
```

The ESP32 can transmit structured JSON sensor readings to the backend.

Example payload:

```json
{
    "device_id": "RETAIL_ESP32_01",
    "visitor_count": 12,
    "distance_cm": 45.5,
    "motion_detected": true,
    "occupancy_status": "NORMAL"
}
```

Example API integration concept:

```python
import requests

url = "http://127.0.0.1:8000/api/sensor-data"

payload = {
    "device_id": "RETAIL_ESP32_01",
    "visitor_count": 12,
    "distance_cm": 45.5,
    "motion_detected": True,
    "occupancy_status": "NORMAL"
}

response = requests.post(url, json=payload)

print(response.status_code)
```

This is an integration example. The corresponding FastAPI endpoint must be implemented before using it.

For a separate ESP32 and laptop connected over Wi-Fi, use the laptop's local network IP address instead of `127.0.0.1`.

---

# 🧠 8. AI Retail Analytics

The platform combines computer vision telemetry and historical retail data to provide business intelligence.

## 8.1 Predictive Purchase Modeling

Uses a Gradient Boosting classifier to estimate purchase probability based on visitor behavior.

Possible input features:

- Zone activity.
- Dwell time.
- Demographic estimates.
- Historical purchase outcomes.

## 8.2 Revenue & Budget Optimization

Uses dynamic programming and a multiple-choice knapsack optimization approach.

The objective is to recommend marketing and store-layout interventions within a fixed budget.

## 8.3 Seven-Day Footfall Forecasting

Uses time-series forecasting approaches such as:

- Seasonal Average.
- Ridge Regression.

These models estimate upcoming visitor traffic and help with staffing decisions.

## 8.4 Customer Segmentation

Uses K-Means clustering to group visitors based on behavioral patterns.

## 8.5 A/B Testing Planner

Provides an A/B testing workflow to calculate statistically valid sample sizes for in-store promotional experiments.

## 8.6 AI Retail Advisor

Integrates Google Gemini to provide:

- Natural-language executive summaries.
- Multi-period performance comparisons.
- Interactive analytics Q&A.
- Retail operational recommendations.

---

# 📊 9. Streamlit Executive Dashboard

The dashboard provides an interactive interface for viewing store analytics.

Key dashboard capabilities include:

### Store Overview

- Total visitors.
- Average dwell time.
- Zone activity.
- Store occupancy.
- Visitor distribution.

### Customer Analytics

- Customer movement.
- Dwell-time analysis.
- Zone engagement.
- Estimated demographic distribution.

### Spatial Heatmap

Displays the generated heatmap to identify customer hotspots and low-traffic areas.

### Predictive Analytics

Displays:

- Purchase probability.
- Footfall forecasts.
- Customer segmentation.
- Revenue optimization.

### Electronics Monitoring

The proposed hardware integration can display:

- ESP32 connection status.
- Live visitor counter.
- IR sensor events.
- Ultrasonic distance.
- PIR motion status.
- Occupancy alerts.

### AI Retail Advisor

Provides business recommendations using the Google Gemini integration.

---

# 🌐 10. FastAPI Backend

The FastAPI backend acts as the communication layer between the processing pipeline, database, hardware integration and dashboard.

Main responsibilities:

- Exposing REST endpoints.
- Retrieving visitor records.
- Providing zone analytics.
- Aggregating store performance.
- Receiving ESP32 sensor data.
- Providing analytics to the dashboard.

Start the backend:

```bash
uvicorn api:app --reload
```

API base URL:

```text
http://127.0.0.1:8000
```

Interactive Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

---

# 🗄️ 11. Database & Data Storage

The system uses SQLite to store retail analytics and visitor telemetry.

### Live Video Database

`video_data.db`

Stores visitor sessions generated by the computer vision pipeline.

### Synthetic Retail Database

`retail_data.db`

Contains generated historical retail data for machine learning, forecasting and analytics.

### Potential Hardware Telemetry

The electronics extension can store:

- Device ID.
- Sensor readings.
- Visitor count.
- Distance measurements.
- Motion events.
- Occupancy status.
- Event timestamps.

---

# 🛠️ 12. Complete Technology Stack

| Category | Technology |
|---|---|
| Programming Language | Python |
| Microcontroller | ESP32 |
| Embedded Programming | Arduino IDE / ESP32 Arduino Core |
| Computer Vision | OpenCV |
| Object Detection | Ultralytics YOLOv8 |
| Object Tracking | DeepSORT |
| Machine Learning | Scikit-learn |
| Numerical Processing | NumPy |
| Data Analysis | Pandas |
| Backend | FastAPI |
| API Server | Uvicorn |
| Database | SQLite |
| ORM | SQLAlchemy |
| Dashboard | Streamlit |
| Visualization | Plotly |
| Generative AI | Google Gemini API |
| Hardware Communication | Wi-Fi / Serial |
| Sensors | IR, HC-SR04, PIR |
| Display | 16x2 LCD with I2C |
| Output Devices | LED and Buzzer |

---

# 💻 13. Hardware Requirements

## Computer / Processing Unit

| Component | Minimum | Recommended |
|---|---|---|
| CPU | Intel Core i5 / Ryzen 5 | Intel Core i7 / Ryzen 7 |
| RAM | 8 GB | 16–32 GB |
| GPU | Integrated graphics | NVIDIA RTX GPU |
| Storage | 10 GB free | 50 GB+ SSD |
| Operating System | Windows 10/11 | Windows 11 / Ubuntu |

A dedicated NVIDIA GPU is recommended for faster computer vision inference, but CPU inference is also possible.

## Camera Requirements

- 1080p Full HD camera.
- Approximately 30 FPS.
- USB webcam or IP CCTV.
- RTSP support for IP camera integration.

## Optional Edge AI Hardware

For future standalone deployment, the system can be adapted to:

- NVIDIA Jetson Orin Nano.
- NVIDIA Jetson Xavier NX.
- Raspberry Pi for selected lightweight sensor and control tasks.

The ESP32 is used for sensor control and does not replace the computer required for full YOLOv8 and DeepSORT processing.

---

# 🚀 14. Quick Start & Installation

## Step 1: Clone the Repository

```bash
git clone https://github.com/saurabhsinghrdjim2-cmd/retail-intelligence-system.git

cd retail-intelligence-system
```

## Step 2: Create a Python Virtual Environment

### Windows PowerShell

```powershell
python -m venv venv

.\venv\Scripts\Activate.ps1
```

### Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

## Step 3: Install Dependencies

```bash
python -m pip install --upgrade pip

pip install -r requirements.txt
```

If the repository does not contain a complete requirements file, install the required packages individually according to the actual project imports.

## Step 4: Configure Environment Variables

If you use Google Gemini, configure your API key.

Example:

```text
GEMINI_API_KEY=your_api_key_here
```

Do not commit API keys or other secrets to GitHub.

---

# ▶️ 15. Running the Software

The repository contains the following main modules:

| File | Responsibility |
|---|---|
| `vision_pipeline.py` | YOLOv8 detection, DeepSORT tracking and movement analytics |
| `dashboard.py` | Streamlit executive dashboard |
| `api.py` | FastAPI backend |
| `generator.py` | Synthetic retail data generation |
| `data_sources.py` | Retail data access |
| `analytics.py` | Analytics processing |
| `advanced.py` | Advanced analytical functionality |
| `insights.py` | Retail insights |
| `gemini_advisor.py` | Gemini AI integration |
| `train_model.py` | Machine learning model training |
| `generate_mock_db.py` | Mock database generation |
| `requirements.txt` | Python dependencies |

### Terminal 1 — Computer Vision Pipeline

```powershell
python vision_pipeline.py
```

If your pipeline requires a video argument, provide the actual video filename:

```powershell
python vision_pipeline.py --video store.mp4 --fresh
```

### Terminal 2 — FastAPI Backend

```powershell
uvicorn api:app --reload --port 8000
```

### Terminal 3 — Streamlit Dashboard

```powershell
streamlit run dashboard.py
```

Dashboard:

```text
http://localhost:8501
```

API:

```text
http://127.0.0.1:8000
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

---

# 🔧 16. Electronics Setup & Execution

## Step 1: Install Arduino IDE

Install Arduino IDE and configure ESP32 board support.

## Step 2: Connect ESP32

Connect the ESP32 to your computer using a USB cable.

## Step 3: Install Required Libraries

Depending on the selected implementation, install:

- LiquidCrystal_I2C.
- ESP32 Arduino Core.
- Required sensor libraries.

## Step 4: Connect Sensors

Connect the IR, HC-SR04 and PIR sensors using the suggested GPIO configuration.

Connect the LCD, LEDs and buzzer.

Follow the electrical safety notes in the hardware section.

## Step 5: Upload ESP32 Firmware

Write and upload firmware to:

- Read sensor values.
- Detect entry and exit sequences.
- Calculate visitor counts.
- Update the LCD.
- Control LEDs.
- Trigger buzzer alerts.
- Send sensor information to the Python backend.

## Step 6: Connect ESP32 to Wi-Fi

Configure the ESP32 with your local Wi-Fi credentials.

The ESP32 and backend computer should be able to communicate over the same network.

## Step 7: Integrate with FastAPI

Implement the sensor-data endpoint in FastAPI and send sensor readings from the ESP32.

## Step 8: Display Sensor Data

Connect the incoming sensor information to the SQLite database and Streamlit dashboard.

---

# 🔄 17. Complete End-to-End Data Flow

```text
                    SMART RETAIL STORE
                            |
                 +----------+----------+
                 |                     |
                 v                     v
             CCTV CAMERA          IR / PIR / HC-SR04
                 |                     |
                 v                     v
              OpenCV                 ESP32
                 |                     |
                 v                     v
              YOLOv8             Sensor Processing
                 |
                 v
              DeepSORT
                 |
                 v
           Customer Tracking
                 |
                 +----------+
                            |
                            v
                     Python Backend
                            |
                   +--------+--------+
                   |                 |
                   v                 v
               SQLite DB       Analytics Engine
                   |                 |
                   +--------+--------+
                            |
                            v
                          FastAPI
                            |
                            v
                    Streamlit Dashboard
                            |
                 +----------+----------+
                 |          |          |
                 v          v          v
                LCD         LED      BUZZER
                            |
                            v
                      STORE MANAGER
```

---

# 🎯 18. Real-World Business Applications

## Store Layout Optimization

Identify high-traffic and low-traffic areas to improve store layout and product placement.

## Customer Engagement

Analyze how long customers spend browsing specific product zones.

## Checkout Monitoring

Identify checkout congestion and monitor average customer dwell time.

## Staffing Optimization

Use historical traffic and predicted visitor counts to plan employee schedules.

## Customer Journey Analysis

Understand how visitors move between different store zones.

## Smart Store Automation

Use electronic sensors to detect entry, movement and occupancy conditions.

## Executive Reporting

Generate retail performance summaries and analytics for store managers.

---

# 🔐 19. Privacy & Responsible AI

This project is intended as a technical retail analytics prototype.

Production deployment should consider:

- Appropriate customer consent and notification.
- Applicable privacy regulations.
- Secure data storage.
- Data retention and deletion policies.
- Access control.
- Model accuracy evaluation.
- Bias and fairness evaluation.
- Appropriate CCTV footage handling.

Demographic classification produces model estimates and should not be treated as verified personal information.

Where possible, minimize the retention of identifiable video and use aggregated analytics.

---

# 📈 20. Future Improvements

Potential improvements include:

### AI & Computer Vision

- Advanced customer behavior analysis.
- Product detection.
- Shelf monitoring.
- Product interaction tracking.
- Real-time anomaly detection.
- Improved customer journey visualization.

### IoT & Electronics

- Multiple ESP32 sensor nodes.
- Wireless sensor communication.
- Additional store occupancy sensors.
- Automated checkout alerts.
- Smart digital signage.
- Automatic lighting control.
- Integration with RFID.
- Energy monitoring.

### Software & Cloud

- Mobile analytics dashboard.
- Cloud-based deployment.
- Real-time notifications through Slack/email.
- LLM-powered retail analytics.
- Historical trend analysis.
- Multi-store analytics.
- TensorRT / ONNX GPU optimization.
- Real-time streaming analytics.

---

# 📂 21. Project Structure

```text
retail-intelligence-system/
│
├── vision_pipeline.py       # Computer vision and visitor tracking
├── dashboard.py             # Streamlit executive dashboard
├── api.py                   # FastAPI backend
├── generator.py             # Synthetic retail data generation
├── data_sources.py          # Data access and sources
├── analytics.py             # Retail analytics
├── advanced.py              # Advanced analytics
├── insights.py              # Operational insights
├── gemini_advisor.py        # Gemini AI integration
├── train_model.py           # Machine learning training
├── generate_mock_db.py      # Mock database generation
│
├── zones.json               # Store zone configuration
├── requirements.txt         # Python dependencies
│
├── video_data.db            # Live visitor data
├── retail_data.db           # Historical/synthetic retail data
├── heatmap_output.png       # Generated movement heatmap
│
├── venv/                    # Python virtual environment
└── README.md                # Project documentation
```

The ESP32 firmware and circuit diagrams can be added in a separate hardware directory as the electronics implementation is developed.

---

# 📜 22. License

This project is licensed under the MIT License.

---

# 👨‍💻 23. Author

**Suryansh Vaish**

GitHub: [suryanshvaish1](https://github.com/suryanshvaish1)

---

# ⭐ If You Find This Project Useful

Consider giving the repository a ⭐ on GitHub!
