# OpenCalValPlan

### 🛰️ OpenCalValPlan: A Reproducible System for Multi-Sensor Calibration, Validation, and SNO Planning

🎯 **Accepted at IGARSS 2026**
https://2026.ieeeigarss.org/

📄 **Accepted Papers List**
https://2026.ieeeigarss.org/papers/accepted_papers.php

**Paper ID: 4025**
**Title:** *OpenCalValPlan: A Reproducible System for Multi-Sensor Calibration, Validation, and SNO Planning*

---

### 👤 Authors

* **Sandeep Kumar Chittimalli** (Corresponding Author)
* Vasala Saicharan
* Suman Bhatta
* Farhad Roni

© 2026 Sandeep Kumar Chittimalli

---

## 🌐 Live Application

👉 **Access the app here:**
https://opencalvalplan.streamlit.app/

---

## 🚀 Overview

OpenCalValPlan is a web-based application built with Streamlit and Google Earth Engine for satellite calibration and validation planning.

The app enables users to analyze past satellite acquisitions, compute Simultaneous Nadir Overpasses (SNOs), and visualize future satellite passes for selected locations and missions.


---

## 🚀 Features

* 🔐 Google OAuth-based authentication
* 🌍 User-based Google Earth Engine integration
* 📡 Past satellite acquisition analysis
* 🛰️ Simultaneous Nadir Overpass (SNO) computation
* 📈 Future satellite pass prediction
* 🗺️ Interactive map visualization
* ☁️ Weather data integration

---

## 🔐 Authentication & Security

* Users sign in with their own Google account
* All Earth Engine operations run under the user's account
* Users must provide their own Google Cloud / Earth Engine project ID
* No credentials or tokens are stored by the application
* Sensitive values are securely managed via Streamlit Secrets

---

## 🧠 Architecture

```
User → Google Login → OAuth Token → Earth Engine (User Project) → Results
```

* Authentication: Google OAuth (server-side)
* Backend: Streamlit
* Data Engine: Google Earth Engine
* Deployment: Streamlit Community Cloud

---

## 👤 How to Use

1. Open the app: https://opencalvalplan.streamlit.app/
2. Click **Connect Google Earth Engine**
3. Sign in with your Google account
4. Enter your Google Cloud / Earth Engine Project ID
5. Click **Submit**
6. Select:

   * Location
   * Date range
   * Satellite missions
7. Run analysis

---

## 🧭 How the App Works (Detailed Guide)

### 📡 What this tool does

* **Past acquisitions (accurate)** from Google Earth Engine

  * Image footprint intersects the selected site buffer

* **Future pass planning (planning-grade)** using TLE + SGP4

  * Requires elevation > 5°
  * Computes closest approach per pass

* **Weather matching (Open-Meteo hourly)**

  * Cloud cover (%)
  * Precipitation (mm)

---

### 🧪 Quality Labeling

* **GOOD** → cloud ≤ 30%, precipitation ≤ 0.2
* **BAD** → cloud ≥ 60% or precipitation ≥ 1.0
* **OK** → otherwise

---

### 🛰️ SNO Candidates

* Past: acquisitions paired within ± time window
* Future: predicted passes paired within ± time window

---

### 🗺️ Map Visualization

* ★ → acquisition or predicted pass
* Ring → part of an SNO pair

---

### 🎛️ Streamlit Interface (Widgets Guide)

* Location selection
* Date range
* Satellite missions
* Site buffer
* SNO time window
* Project ID input
* Submit button

---

### ⚙️ Processing Flow

```
User Input → Authentication → EE Query / Orbit Propagation → Weather Matching → SNO Pairing → Visualization
```

---

## 📊 Key Capabilities

* Past acquisition retrieval
* Multi-sensor SNO detection
* Future pass prediction
* Interactive geospatial visualization

---

## ⚠️ Requirements

* Google account
* Earth Engine access
* Google Cloud project

---

## 🔒 Privacy & Usage

* This app does not store user credentials
* All processing runs under the user's account
* Users are responsible for their own project usage and billing
* Do not share sensitive information

---


---

## 📄 License

MIT License

---

## 🙌 Acknowledgements

* Google Earth Engine
* Streamlit
* IGARSS 2026 Community

---


