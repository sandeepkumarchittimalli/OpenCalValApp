from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, date
from math import radians, sin, cos, acos
from typing import Dict, Any, List, Optional, Tuple

import time
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

import ee
from itsdangerous import URLSafeSerializer, BadSignature
from google.oauth2.credentials import Credentials
from pyorbital.orbital import Orbital

# Skyfield optional (future)
try:
    from skyfield.api import load, EarthSatellite, wgs84, utc
    SKYFIELD_AVAILABLE = True
except Exception:
    SKYFIELD_AVAILABLE = False

import folium
from folium.plugins import MousePosition, Draw, MarkerCluster
from folium.features import DivIcon
from streamlit_folium import st_folium




st.set_page_config(layout="wide")
st.title(
    "OpenCalValPlan: A Reproducible System for Multi-Sensor Calibration, Validation, and SNO Planning")
# ------------------- USER SETTINGS -------------------

with st.expander("📄 About / Citation"):
    st.markdown("""
**OpenCalValPlan: A Reproducible System for Multi-Sensor Calibration, Validation, and SNO Planning** 

🎯 Accepted at **IGARSS 2026** https://2026.ieeeigarss.org/

📄 Accepted Papers List:  
https://2026.ieeeigarss.org/papers/accepted_papers.php  

4025	OPENCALVALPLAN: A REPRODUCIBLE SYSTEM FOR MULTI-SENSOR CALIBRATION, VALIDATION, AND SNO PLANNING

📄 Corresponding Author  
[Sandeep Kumar Chittimalli](https://sandeepkumarchittimalli.github.io)

📄 Co-authors  
Vasala Saicharan,  
Suman Bhatta,  
Farhad Roni  


© 2026 Sandeep Kumar Chittimalli
""")


DEFAULT_LAT = 42.944611
DEFAULT_LON = -122.109245

DEFAULT_OVERPASS_TOL_KM = 50.0
DEFAULT_SITE_BUFFER_M = 60.0  # past acquisition at site = footprint intersects this buffer

S2_TOA_SCALE = 1.0 / 10000.0

GOOD_CLOUD_PCT = 30.0
BAD_CLOUD_PCT = 60.0
GOOD_PRECIP_MM = 0.2
BAD_PRECIP_MM = 1.0

MIN_ELEV_DEG_DEFAULT = 5.0  # future pass planning threshold


ABOUT_MD = f"""
### What this tool does

- **Past acquisitions (accurate)** from **Google Earth Engine**  
  “Acquisition at site” = image footprint intersects the **site buffer** around the selected point.

- **Future pass planning (planning-grade)** from **TLE + SGP4**  
  Requires satellite above horizon (elevation > {MIN_ELEV_DEG_DEFAULT:.0f}° default).  
  Finds closest approach per visible pass.

- **Weather matched to acquisition/pass time (Open-Meteo hourly)**  
  cloud cover (%), precipitation (mm)

- **Quality label**
  - GOOD if cloud ≤ {GOOD_CLOUD_PCT:.0f}% and precip ≤ {GOOD_PRECIP_MM}
  - BAD if cloud ≥ {BAD_CLOUD_PCT:.0f}% or precip ≥ {BAD_PRECIP_MM}
  - otherwise OK

- **SNO candidates (all mission pairs)**
  - Past: acquisitions paired within ± window
  - Future: predicted passes paired within ± window

✅ Results are shown directly on the **same map**:  
- **★** = acquisition / pass  
- **ring** = part of an SNO pair

### SNO time window guidance

Different satellites were flown with very different orbital strategies.  
The SNO time window should be chosen based on the satellite pair:

**True constellation SNOs (use short windows, 30–60 minutes):**
- Landsat 8 ↔ Landsat 9  
- Sentinel-2A ↔ Sentinel-2B  

**Near-simultaneous collocations (use 1–3 hours):**
- Landsat ↔ Sentinel-2  

**Wide-swath collocations (use 6–12 hours):**
- Landsat ↔ MODIS (Terra/Aqua)  
- Landsat ↔ VIIRS (Suomi-NPP)  

**Early Landsat era (use long windows, 12–24 hours up to multi-day):**
- Landsat-1 / 2 / 3 (MSS)  
- Landsat-4 / 5 (MSS or TM)

Choosing too short a window for early missions will correctly return zero SNOs.
"""


# Quick test sites (Sonoran corrected to be inside desert region)
TEST_SITES = {
    "Custom (use inputs)": None,
    "Crater Lake (OR)": (42.944611, -122.109245),
    "Sonoran Desert (AZ) – center-ish": (31.8500823, -111.8568586),
    "Algodones Dunes (CA)": (32.9583762, -115.0841386),
    "Lake Tahoe (CA/NV)": (39.096848, -120.032349),
    "Libya-4 PICS (Libya)": (28.55, 23.39),
    "Railroad Valley Playa (NV)": (38.497, -115.690),
    "Niger-1 (CEOS PICS)": (20.41, 9.36),
    "Libya-3 (PICS)": (23.15, 23.10),
    "Algeria-5 (PICS)": (31.02, 2.23),
    "Algeria-3 (PICS)": (30.32, 7.66),
    "Mauritania-1 (PICS)": (19.40, -9.30),
    "Mauritania-2 (PICS)": (20.85, -8.78),
    "Niger-2 (PICS)": (21.08, 10.44),
    "Namib Desert-1 (PICS)": (-24.98, 15.27),
    "Namib Desert-2 (PICS)": (-17.33, 12.05),
}


# Bright high-contrast colors for stars
MISSION_COLORS: Dict[str, str] = {
    "LANDSAT 1 MSS": "#F500FF",
    "LANDSAT 2 MSS": "#00E5FF",
    "LANDSAT 3 MSS": "#FF1744",
    "LANDSAT 4 MSS": "#FF9100",
    "LANDSAT 5 MSS": "#FFEA00",
    "LANDSAT 4 TM": "#76FF03",
    "LANDSAT 5 TM": "#00E676",
    "LANDSAT 7": "#FFD43B",
    "LANDSAT 8": "#FF0000",
    "LANDSAT 9": "#00FFFF",  # changed from white for visibility across layers

    "SENTINEL-2A": "#00FFFF",
    "SENTINEL-2B": "#2979FF",
    "SENTINEL-1A": "#7C4DFF",
    "SENTINEL-3A": "#FF6D00",
    "SENTINEL-3B": "#FF4081",

    "TERRA (MODIS)": "#7CFF00",
    "AQUA (MODIS)": "#00FF7F",
    "SUOMI NPP (VIIRS)": "#FF00FF",
}


# ------------------- FUTURE SATELLITES (TLE) -------------------
# Landsat 7 is decommissioned (keep for past, omit from future)
SATELLITE_NORAD: Dict[str, int] = {
    "LANDSAT 8": 39084,
    "LANDSAT 9": 49260,

    "SENTINEL-2A": 40697,
    "SENTINEL-2B": 42063,
    "SENTINEL-1A": 39634,
    "SENTINEL-3A": 41335,
    "SENTINEL-3B": 43437,

    "TERRA (MODIS)": 25994,
    "AQUA (MODIS)": 27424,
    "SUOMI NPP (VIIRS)": 37849,
}


# ------------------- PAST MISSIONS (GEE) -------------------

@dataclass(frozen=True)
class PastMission:
    key: str
    label: str
    collections: Tuple[str, ...]
    id_prop: Optional[str]
    cloud_prop: Optional[str]
    sun_az_prop: Optional[str]
    sun_zen_prop: Optional[str]
    spacecraft_prop: Optional[str]
    spacecraft_values: Optional[Tuple[str, ...]]
    sample_scale_m: Optional[int]
    sample_bands: Optional[Tuple[str, ...]]
    sample_band_scales: Optional[Tuple[float, ...]]


PAST_MISSIONS: Dict[str, PastMission] = {
    # MSS (include T1 + T2)
    "LANDSAT 1 MSS": PastMission(
        key="LANDSAT 1 MSS", label="Landsat 1 (MSS)",
        collections=("LANDSAT/LM01/C02/T1", "LANDSAT/LM01/C02/T2"),
        id_prop=None, cloud_prop=None, sun_az_prop=None, sun_zen_prop=None,
        spacecraft_prop=None, spacecraft_values=None,
        sample_scale_m=None, sample_bands=None, sample_band_scales=None
    ),
    "LANDSAT 2 MSS": PastMission(
        key="LANDSAT 2 MSS", label="Landsat 2 (MSS)",
        collections=("LANDSAT/LM02/C02/T1", "LANDSAT/LM02/C02/T2"),
        id_prop=None, cloud_prop=None, sun_az_prop=None, sun_zen_prop=None,
        spacecraft_prop=None, spacecraft_values=None,
        sample_scale_m=None, sample_bands=None, sample_band_scales=None
    ),
    "LANDSAT 3 MSS": PastMission(
        key="LANDSAT 3 MSS", label="Landsat 3 (MSS)",
        collections=("LANDSAT/LM03/C02/T1", "LANDSAT/LM03/C02/T2"),
        id_prop=None, cloud_prop=None, sun_az_prop=None, sun_zen_prop=None,
        spacecraft_prop=None, spacecraft_values=None,
        sample_scale_m=None, sample_bands=None, sample_band_scales=None
    ),
    "LANDSAT 4 MSS": PastMission(
        key="LANDSAT 4 MSS", label="Landsat 4 (MSS)",
        collections=("LANDSAT/LM04/C02/T1", "LANDSAT/LM04/C02/T2"),
        id_prop=None, cloud_prop=None, sun_az_prop=None, sun_zen_prop=None,
        spacecraft_prop=None, spacecraft_values=None,
        sample_scale_m=None, sample_bands=None, sample_band_scales=None
    ),
    "LANDSAT 5 MSS": PastMission(
        key="LANDSAT 5 MSS", label="Landsat 5 (MSS)",
        collections=("LANDSAT/LM05/C02/T1", "LANDSAT/LM05/C02/T2"),
        id_prop=None, cloud_prop=None, sun_az_prop=None, sun_zen_prop=None,
        spacecraft_prop=None, spacecraft_values=None,
        sample_scale_m=None, sample_bands=None, sample_band_scales=None
    ),

    # TM/ETM+/OLI TOA
    "LANDSAT 4 TM": PastMission(
        key="LANDSAT 4 TM", label="Landsat 4 (TM) TOA",
        collections=("LANDSAT/LT04/C02/T1_TOA",),
        id_prop="LANDSAT_PRODUCT_ID", cloud_prop="CLOUD_COVER",
        sun_az_prop="SUN_AZIMUTH", sun_zen_prop=None,
        spacecraft_prop="SPACECRAFT_ID", spacecraft_values=("LANDSAT_4",),
        sample_scale_m=None, sample_bands=None, sample_band_scales=None
    ),
    "LANDSAT 5 TM": PastMission(
        key="LANDSAT 5 TM", label="Landsat 5 (TM) TOA",
        collections=("LANDSAT/LT05/C02/T1_TOA",),
        id_prop="LANDSAT_PRODUCT_ID", cloud_prop="CLOUD_COVER",
        sun_az_prop="SUN_AZIMUTH", sun_zen_prop=None,
        spacecraft_prop="SPACECRAFT_ID", spacecraft_values=("LANDSAT_5",),
        sample_scale_m=None, sample_bands=None, sample_band_scales=None
    ),
    "LANDSAT 7": PastMission(
        key="LANDSAT 7", label="Landsat 7 (ETM+) TOA (decommissioned, past data exists)",
        collections=("LANDSAT/LE07/C02/T1_TOA",),
        id_prop="LANDSAT_PRODUCT_ID", cloud_prop="CLOUD_COVER",
        sun_az_prop="SUN_AZIMUTH", sun_zen_prop=None,
        spacecraft_prop="SPACECRAFT_ID", spacecraft_values=("LANDSAT_7",),
        sample_scale_m=None, sample_bands=None, sample_band_scales=None
    ),
    "LANDSAT 8": PastMission(
        key="LANDSAT 8", label="Landsat 8 (OLI) TOA",
        collections=("LANDSAT/LC08/C02/T1_TOA",),
        id_prop="LANDSAT_PRODUCT_ID", cloud_prop="CLOUD_COVER",
        sun_az_prop="SUN_AZIMUTH", sun_zen_prop=None,
        spacecraft_prop="SPACECRAFT_ID", spacecraft_values=("LANDSAT_8",),
        sample_scale_m=None, sample_bands=None, sample_band_scales=None
    ),
    "LANDSAT 9": PastMission(
        key="LANDSAT 9", label="Landsat 9 (OLI-2) TOA",
        collections=("LANDSAT/LC09/C02/T1_TOA",),
        id_prop="LANDSAT_PRODUCT_ID", cloud_prop="CLOUD_COVER",
        sun_az_prop="SUN_AZIMUTH", sun_zen_prop=None,
        spacecraft_prop="SPACECRAFT_ID", spacecraft_values=("LANDSAT_9",),
        sample_scale_m=None, sample_bands=None, sample_band_scales=None
    ),

    # Sentinel-2 TOA
    "SENTINEL-2A": PastMission(
        key="SENTINEL-2A", label="Sentinel-2A TOA",
        collections=("COPERNICUS/S2_HARMONIZED",),
        id_prop="PRODUCT_ID", cloud_prop="CLOUDY_PIXEL_PERCENTAGE",
        sun_az_prop="MEAN_SOLAR_AZIMUTH_ANGLE", sun_zen_prop="MEAN_SOLAR_ZENITH_ANGLE",
        spacecraft_prop="SPACECRAFT_NAME", spacecraft_values=("Sentinel-2A",),
        sample_scale_m=None, sample_bands=None, sample_band_scales=None
    ),
    "SENTINEL-2B": PastMission(
        key="SENTINEL-2B", label="Sentinel-2B TOA",
        collections=("COPERNICUS/S2_HARMONIZED",),
        id_prop="PRODUCT_ID", cloud_prop="CLOUDY_PIXEL_PERCENTAGE",
        sun_az_prop="MEAN_SOLAR_AZIMUTH_ANGLE", sun_zen_prop="MEAN_SOLAR_ZENITH_ANGLE",
        spacecraft_prop="SPACECRAFT_NAME", spacecraft_values=("Sentinel-2B",),
        sample_scale_m=None, sample_bands=None, sample_band_scales=None
    ),

    # MODIS
    "TERRA (MODIS)": PastMission(
        key="TERRA (MODIS)", label="Terra MODIS SR (daily, 250m)",
        collections=("MODIS/061/MOD09GQ",),
        id_prop=None, cloud_prop=None,
        sun_az_prop=None, sun_zen_prop=None,
        spacecraft_prop=None, spacecraft_values=None,
        sample_scale_m=250, sample_bands=("sur_refl_b01", "sur_refl_b02"),
        sample_band_scales=(0.0001, 0.0001)
    ),
    "AQUA (MODIS)": PastMission(
        key="AQUA (MODIS)", label="Aqua MODIS SR (daily, 250m)",
        collections=("MODIS/061/MYD09GQ",),
        id_prop=None, cloud_prop=None,
        sun_az_prop=None, sun_zen_prop=None,
        spacecraft_prop=None, spacecraft_values=None,
        sample_scale_m=250, sample_bands=("sur_refl_b01", "sur_refl_b02"),
        sample_band_scales=(0.0001, 0.0001)
    ),

    # VIIRS
    "SUOMI NPP (VIIRS)": PastMission(
        key="SUOMI NPP (VIIRS)", label="Suomi NPP VIIRS SR (daily)",
        collections=("NASA/VIIRS/002/VNP09GA",),
        id_prop=None, cloud_prop=None,
        sun_az_prop=None, sun_zen_prop=None,
        spacecraft_prop=None, spacecraft_values=None,
        sample_scale_m=500, sample_bands=("I1", "I2"),
        sample_band_scales=(0.0001, 0.0001)
    ),
}

#------------------- GOOGLE OAUTH (CLOUD RUN) + GEE INIT -------------------

OAUTH_BACKEND_START = st.secrets["google_oauth"]["oauth_backend_start"]
SIGNING_SECRET = st.secrets["google_oauth"]["signing_secret"]
serializer = URLSafeSerializer(SIGNING_SECRET, salt="oauth-return")

def finish_oauth_if_needed():
    qp = st.query_params
    signed = qp.get("oauth_return")
    if not signed:
        return

    try:
        payload = serializer.loads(signed)
    except BadSignature:
        st.error("OAuth return invalid.")
        st.stop()

    st.session_state["google_tokens"] = payload
    st.query_params.clear()
    st.rerun()

def get_user_google_credentials() -> Optional[Credentials]:
    data = st.session_state.get("google_tokens")
    if not data:
        return None

    return Credentials(
        token=data["token"],
        refresh_token=data.get("refresh_token"),
        token_uri=data["token_uri"],
        client_id=data["client_id"],
        client_secret=data["client_secret"],
        scopes=data["scopes"],
    )

def init_ee(project_id: str) -> str:
    creds = get_user_google_credentials()
    if creds is None:
        raise RuntimeError("User is not connected to Google Earth Engine.")
    ee.Initialize(credentials=creds, project=project_id)
    return f"user_oauth(project={project_id})"

finish_oauth_if_needed()

st.sidebar.header("Google Earth Engine Login")
st.sidebar.caption("🔐 Uses your Google account and project for Earth Engine processing")
st.sidebar.info(
    "Sign in with your Google account, then enter your own Earth Engine project ID. "
    "Please review Privacy & Usage before using this app."
)
st.sidebar.markdown(
    "📖 [Official GEE Setup Guide](https://developers.google.com/earth-engine/guides/auth)"
)

if "google_tokens" not in st.session_state:
    st.sidebar.link_button("Connect Google Earth Engine", OAUTH_BACKEND_START)
    st.sidebar.warning("Connect your Google account first.")
    st.stop()

st.sidebar.success("Google account connected ✅")

# ------------------- PROJECT ID -------------------
if "project_submitted" not in st.session_state:
    st.session_state["project_submitted"] = False
if "submitted_project_id" not in st.session_state:
    st.session_state["submitted_project_id"] = ""
if "gee_project_id_input" not in st.session_state:
    st.session_state["gee_project_id_input"] = st.session_state["submitted_project_id"]

st.sidebar.text_input(
    "Enter your GEE Project ID",
    value=st.session_state.get("gee_project_id_input", st.session_state["submitted_project_id"]),
    help="Provide your own Google Earth Engine Project ID (e.g., my-project-123)",
    key="gee_project_id_input",
)
submit_project = st.sidebar.button("Submit Project ID", key="submit_project_id_btn")

if submit_project:
    cleaned_project_id = st.session_state.get("gee_project_id_input", "").strip()
    if cleaned_project_id:
        st.session_state["submitted_project_id"] = cleaned_project_id
        st.session_state["project_submitted"] = True
        st.rerun()
    else:
        st.session_state["project_submitted"] = False
        st.sidebar.warning("Please enter a valid GEE Project ID.")
        st.stop()

if not st.session_state["project_submitted"]:
    st.sidebar.info("Enter your GEE project ID and click Submit Project ID.")
    st.stop()

project_id = st.session_state["submitted_project_id"].strip()
if not project_id:
    st.sidebar.warning("Please enter a valid GEE Project ID.")
    st.stop()

if st.sidebar.button("Change Project ID", key="change_project_id_btn"):
    st.session_state["project_submitted"] = False
    st.session_state["submitted_project_id"] = ""
    st.session_state["gee_project_id_input"] = ""
    st.rerun()

try:
    ee_status = init_ee(project_id)
    st.sidebar.success(f"GEE initialized successfully ✅ ({project_id})")
except Exception as e:
    st.sidebar.error("GEE initialization failed ❌")
    st.sidebar.write(str(e))
    st.stop()

RESULT_STATE_KEYS = [
    "past_df_raw", "past_df", "past_sno", "past_selected",
    "future_df_raw", "future_df", "future_sno", "future_selected",
    "runtime_s", "last_compute_params",
]

def clear_results_state() -> None:
    for key in RESULT_STATE_KEYS:
        st.session_state.pop(key, None)


def mark_results_dirty() -> None:
    st.session_state["results_dirty"] = True
    st.session_state["dirty_reason"] = "params"


def reset_app_state() -> None:
    clear_results_state()
    st.session_state["results_dirty"] = False
    st.session_state["lat"] = float(DEFAULT_LAT)
    st.session_state["lon"] = float(DEFAULT_LON)
    st.session_state["lat_input"] = float(DEFAULT_LAT)
    st.session_state["lon_input"] = float(DEFAULT_LON)
    st.session_state["_last_click"] = None
    st.session_state["mode"] = "Past acquisitions"
    today = datetime.utcnow().date()
    yesterday = today - timedelta(days=1)
    st.session_state["date_range"] = (today - timedelta(days=90), yesterday)
    st.session_state["site_choice"] = "Crater Lake (OR)"
    st.session_state["_site_choice_applied"] = "Crater Lake (OR)"
    st.session_state["map_view_center"] = [float(DEFAULT_LAT), float(DEFAULT_LON)]
    st.session_state["map_view_zoom"] = 7
    st.session_state["selected_past"] = ["LANDSAT 8", "LANDSAT 9", "SENTINEL-2A", "SENTINEL-2B"]
    st.session_state["selected_future"] = ["LANDSAT 8", "LANDSAT 9", "SENTINEL-2A", "SENTINEL-2B"]
    st.session_state["sno_window_choice"] = "30 min"
    st.session_state.pop("sno_window_custom", None)
    st.session_state["site_buffer_m"] = int(DEFAULT_SITE_BUFFER_M)
    st.session_state["sample_reflectance"] = False
    st.session_state["overpass_tol_km"] = float(DEFAULT_OVERPASS_TOL_KM)
    st.session_state["min_elev_deg"] = float(MIN_ELEV_DEG_DEFAULT)

# ------------------- UTILS -------------------

def great_circle_distance_km(lat1_deg: float, lon1_deg: float, lat2_deg: float, lon2_deg: float) -> float:
    R = 6378.0
    lat1 = radians(lat1_deg); lon1 = radians(lon1_deg)
    lat2 = radians(lat2_deg); lon2 = radians(lon2_deg)
    return R * acos(sin(lat1)*sin(lat2) + cos(lat1)*cos(lat2)*cos(lon1-lon2))


def _safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def classify_conditions(cloud_pct: Optional[float], precip_mm: Optional[float]) -> str:
    if cloud_pct is None and precip_mm is None:
        return "Unknown"
    c = float(cloud_pct) if cloud_pct is not None and not pd.isna(cloud_pct) else 999.0
    p = float(precip_mm) if precip_mm is not None and not pd.isna(precip_mm) else 999.0
    if c <= GOOD_CLOUD_PCT and p <= GOOD_PRECIP_MM:
        return "GOOD"
    if c >= BAD_CLOUD_PCT or p >= BAD_PRECIP_MM:
        return "BAD"
    return "OK"


def style_quality_rows(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    colors = {
        "GOOD": "background-color: #d8f3dc; color: #000;",
        "OK": "background-color: #fff3bf; color: #000;",
        "BAD": "background-color: #ffd6d6; color: #000;",
        "Unknown": "background-color: #e9ecef; color: #000;",
    }
    def row_style(row):
        label = row.get("quality_label", "Unknown")
        css = colors.get(label, colors["Unknown"])
        return [css] * len(row)
    return df.style.apply(row_style, axis=1)


def mission_hex_color(mission_key: str) -> str:
    return MISSION_COLORS.get(mission_key, "#00E5FF")


def normalize_time_to_sec(t: Any) -> pd.Timestamp:
    ts = pd.to_datetime(t, utc=False)
    return ts.floor("S")


def format_runtime(seconds: float) -> str:
    seconds = float(max(0.0, seconds))
    mins = int(seconds // 60)
    secs = seconds - mins * 60
    return f"{mins}m {secs:.1f}s"


def format_dt_minutes(dt_min: float) -> str:
    try:
        dt_min = float(dt_min)
    except Exception:
        return ""
    if dt_min < 60:
        return f"{dt_min:.2f} min"
    dt_hr = dt_min / 60.0
    if dt_hr < 24:
        return f"{dt_hr:.2f} hr"
    dt_days = dt_hr / 24.0
    return f"{dt_days:.2f} days"


def add_map_runtime_overlay(m: folium.Map, runtime_s: float) -> None:
    rt_txt = format_runtime(runtime_s)
    runtime_html = f"""
    <div style="position: fixed; top: 14px; right: 14px; z-index: 9999;
         background: rgba(11, 19, 32, 0.95);
         padding: 12px 14px; border-radius: 12px;
         border: 2px solid #ff6b6b; box-shadow: 0 0 12px rgba(255,107,107,0.35);">
      <div style="font-size: 13px; font-weight: 700; color: #ffd8a8; letter-spacing: 0.3px;">
        LAST COMPUTE TIME
      </div>
      <div style="font-size: 24px; font-weight: 900; color: #ff4d4f; line-height: 1.2;">
        {rt_txt}
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(runtime_html))


def add_map_legend_overlay(m: folium.Map, missions: List[str]) -> None:
    if not missions:
        return
    mission_lines = "".join(
        [
            f"<div style='margin: 4px 0; font-size: 13px; font-weight: 700;'>"
            f"<span style='display:inline-block; width:18px; text-align:center; color:{mission_hex_color(s)}; "
            f"font-size:18px; font-weight:900;'>★</span> {s}</div>"
            for s in missions
        ]
    )
    legend_html = f"""
    <div style="position: fixed; bottom: 26px; left: 14px; z-index: 9999;
      background: rgba(11, 19, 32, 0.95); color: #f8f9fa;
      padding: 12px 14px; border: 1px solid #334155; border-radius: 12px;
      box-shadow: 0 0 12px rgba(0,0,0,0.28); min-width: 210px;">
      <div style="font-size: 14px; font-weight: 900; color: #ffffff; margin-bottom: 6px;">Legend</div>
      {mission_lines}
      <div style="margin-top: 8px; font-size: 13px; font-weight: 700;">
        <span style="display:inline-block; width:12px; height:12px; border:3px solid #FFD43B;
        border-radius:50%; margin-right:8px; vertical-align:middle;"></span>SNO ring
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))


def add_star(layer, lat, lon, color_hex="#00FFFF", tooltip="", popup_html=""):
    """
    High-visibility star marker:
    - larger size
    - thick dark outline
    - white glow halo
    """
    html = f"""
    <div style="
        font-size: 22px;
        font-weight: 900;
        color: {color_hex};
        line-height: 22px;
        -webkit-text-stroke: 2px rgba(0,0,0,0.95);
        text-shadow:
            0 0 2px rgba(255,255,255,0.95),
            0 0 6px rgba(255,255,255,0.85),
            0 0 10px rgba(0,0,0,0.60);
        ">
        ★
    </div>"""
    folium.Marker(
        location=[lat, lon],
        tooltip=tooltip,
        popup=folium.Popup(popup_html, max_width=450) if popup_html else None,
        icon=DivIcon(icon_size=(22, 22), icon_anchor=(11, 11), html=html),
    ).add_to(layer)


def add_sno_ring(layer, lat, lon, color="#FFD43B"):
    folium.CircleMarker(
        location=[lat, lon],
        radius=10,
        color=color,
        weight=3,
        fill=False,
        opacity=0.95
    ).add_to(layer)


# ------------------- WEATHER -------------------

@st.cache_data(show_spinner=False)
def fetch_hourly_weather(lat: float, lon: float, start_d: date, end_d: date) -> pd.DataFrame:
    today = datetime.utcnow().date()

    def fetch_range(base_url: str, s: date, e: date) -> List[Dict[str, Any]]:
        url = (
            f"{base_url}"
            f"?latitude={lat}&longitude={lon}"
            "&hourly=cloudcover,precipitation"
            f"&start_date={s.isoformat()}&end_date={e.isoformat()}"
            "&timezone=UTC"
        )
        r = requests.get(url, timeout=25)
        r.raise_for_status()
        data = r.json()
        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        clouds = hourly.get("cloudcover", [])
        precip = hourly.get("precipitation", [])
        out = []
        for t, c, p in zip(times, clouds, precip):
            try:
                dt = datetime.fromisoformat(t)
                out.append({"time": dt, "cloud_cover_pct": float(c), "precip_mm": float(p)})
            except Exception:
                continue
        return out

    records: List[Dict[str, Any]] = []

    archive_end = min(end_d, today - timedelta(days=2))
    if start_d <= archive_end:
        try:
            records += fetch_range("https://archive-api.open-meteo.com/v1/archive", start_d, archive_end)
        except Exception:
            pass

    forecast_start = max(start_d, today - timedelta(days=1))
    forecast_end = min(end_d, today + timedelta(days=14))
    if forecast_start <= forecast_end:
        try:
            records += fetch_range("https://api.open-meteo.com/v1/forecast", forecast_start, forecast_end)
        except Exception:
            pass

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records).drop_duplicates(subset=["time"]).sort_values("time")


def attach_weather(df_events: pd.DataFrame, df_hourly: pd.DataFrame) -> pd.DataFrame:
    if df_events.empty:
        return df_events

    out = df_events.copy()
    out["time"] = pd.to_datetime(out["time"])

    if df_hourly is None or df_hourly.empty:
        out["weather_time"] = pd.NaT
        out["cloud_cover_pct"] = np.nan
        out["precip_mm"] = np.nan
        out["quality_label"] = "Unknown"
        return out

    w = df_hourly.copy()
    w["time"] = pd.to_datetime(w["time"])
    t_hour = w["time"].values.astype("datetime64[ns]")
    clouds = w["cloud_cover_pct"].values
    precip = w["precip_mm"].values

    def nearest_idx(ts: np.datetime64) -> int:
        i = np.searchsorted(t_hour, ts)
        if i <= 0:
            return 0
        if i >= len(t_hour):
            return len(t_hour) - 1
        before = i - 1
        after = i
        d_before = abs((ts - t_hour[before]).astype("timedelta64[s]").astype(int))
        d_after = abs((t_hour[after] - ts).astype("timedelta64[s]").astype(int))
        return before if d_before <= d_after else after

    idxs = [nearest_idx(t) for t in out["time"].values.astype("datetime64[ns]")]
    out["weather_time"] = w.iloc[idxs]["time"].values
    out["cloud_cover_pct"] = clouds[idxs]
    out["precip_mm"] = precip[idxs]
    out["quality_label"] = [
        classify_conditions(c, p) for c, p in zip(out["cloud_cover_pct"], out["precip_mm"])
    ]
    return out


# ------------------- GEE HELPERS -------------------

def _ee_point(lon: float, lat: float) -> ee.Geometry:
    return ee.Geometry.Point([lon, lat])

def _ee_buffer(lon: float, lat: float, meters: float) -> ee.Geometry:
    return _ee_point(lon, lat).buffer(meters)

def _mean_sample(img: ee.Image, region: ee.Geometry, bands: List[str], scale: int) -> ee.Dictionary:
    return img.select(bands).reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=region,
        scale=scale,
        bestEffort=True,
        maxPixels=1_000_000
    )

def _add_centroid_safe(img: ee.Image, pt: ee.Geometry) -> ee.Image:
    """
    Robust centroid extraction:
    Some products yield centroid.coordinates() as an empty list.
    Guard against it to avoid 'List.get: List is empty' in map().
    """
    geom = img.geometry()
    centroid = geom.centroid(1)

    coords = centroid.coordinates()
    pt_coords = pt.coordinates()

    lon = ee.Number(ee.Algorithms.If(coords.length().gte(2), coords.get(0), pt_coords.get(0)))
    lat = ee.Number(ee.Algorithms.If(coords.length().gte(2), coords.get(1), pt_coords.get(1)))

    centroid_safe = ee.Geometry.Point([lon, lat])
    dist_km = centroid_safe.distance(pt).divide(1000.0)

    return img.set({
        "scene_center_dist_km": dist_km,
        "scene_center_lon": lon,
        "scene_center_lat": lat,
    })


@st.cache_data(show_spinner=True)
def gee_past_acquisitions(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    selected_missions: Tuple[str, ...],
    site_buffer_m: float,
    sample_reflectance: bool,
    max_results_per_collection: int = 5000,  # avoid EE abort
) -> pd.DataFrame:
    init_ee(st.session_state["submitted_project_id"])

    pt = _ee_point(lon, lat)
    site = _ee_buffer(lon, lat, float(site_buffer_m))

    rows: List[Dict[str, Any]] = []

    for key in selected_missions:
        mission = PAST_MISSIONS.get(key)
        if mission is None:
            continue

        for col_id in mission.collections:
            ic = ee.ImageCollection(col_id).filterDate(start_date, end_date).filterBounds(site)
            if mission.spacecraft_prop and mission.spacecraft_values:
                ic = ic.filter(ee.Filter.inList(mission.spacecraft_prop, list(mission.spacecraft_values)))

            ic = ic.map(lambda img: _add_centroid_safe(img, pt))
            ic = ic.limit(int(max_results_per_collection))

            def to_feat(img: ee.Image) -> ee.Feature:
                t = img.date().format("YYYY-MM-dd HH:mm:ss")
                scene_id = img.get(mission.id_prop) if mission.id_prop else img.get("system:index")
                cloud = img.get(mission.cloud_prop) if mission.cloud_prop else None
                sun_az = img.get(mission.sun_az_prop) if mission.sun_az_prop else None
                sun_zen = img.get(mission.sun_zen_prop) if mission.sun_zen_prop else None

                sun_el = img.get("SUN_ELEVATION")
                sun_zen2 = ee.Algorithms.If(sun_zen, sun_zen,
                                           ee.Algorithms.If(sun_el, ee.Number(90).subtract(ee.Number(sun_el)), None))

                props = {
                    "time": t,
                    "sat_name": mission.key,
                    "mission": mission.key,
                    "collection": col_id,
                    "scene_id": scene_id,
                    "cloud_scene_pct": cloud,
                    "sun_azimuth_deg": sun_az,
                    "sun_zenith_deg": sun_zen2,
                    "scene_center_dist_km": img.get("scene_center_dist_km"),
                    "scene_center_lat": img.get("scene_center_lat"),
                    "scene_center_lon": img.get("scene_center_lon"),
                }

                if sample_reflectance and mission.sample_bands and mission.sample_scale_m:
                    vals = _mean_sample(img, site, list(mission.sample_bands), scale=int(mission.sample_scale_m))
                    for b in mission.sample_bands:
                        props[f"sample_{b}"] = vals.get(b)

                return ee.Feature(None, props)

            feats = ee.FeatureCollection(ic.map(to_feat)).getInfo().get("features", [])
            for f in feats:
                p = f.get("properties", {})
                t_str = p.get("time")
                if not t_str:
                    continue
                try:
                    t = datetime.strptime(t_str, "%Y-%m-%d %H:%M:%S")
                except Exception:
                    continue

                row: Dict[str, Any] = {
                    "time": t,
                    "sat_name": p.get("sat_name"),
                    "mission": p.get("mission"),
                    "collection": p.get("collection"),
                    "scene_id": p.get("scene_id"),
                    "cloud_scene_pct": _safe_float(p.get("cloud_scene_pct")),
                    "sun_azimuth_deg": _safe_float(p.get("sun_azimuth_deg")),
                    "sun_zenith_deg": _safe_float(p.get("sun_zenith_deg")),
                    "scene_center_dist_km": _safe_float(p.get("scene_center_dist_km")),
                    "scene_center_lat": _safe_float(p.get("scene_center_lat")),
                    "scene_center_lon": _safe_float(p.get("scene_center_lon")),
                }

                if sample_reflectance and mission.sample_bands and mission.sample_scale_m:
                    for i, b in enumerate(mission.sample_bands):
                        v = _safe_float(p.get(f"sample_{b}"))
                        if v is None:
                            row[f"refl_{b}"] = np.nan
                        else:
                            scale = mission.sample_band_scales[i] if mission.sample_band_scales else 1.0
                            row[f"refl_{b}"] = float(v) * float(scale)

                rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values(["time", "sat_name"]).reset_index(drop=True)
    for c in ["scene_center_lat", "scene_center_lon"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ------------------- SNO (PAST) -------------------

def compute_snos_allpairs(df_events: pd.DataFrame, sno_window_minutes: float) -> pd.DataFrame:
    if df_events is None or df_events.empty:
        return pd.DataFrame()

    df = df_events.copy()
    df["time"] = pd.to_datetime(df["time"]).apply(normalize_time_to_sec)
    df = df.sort_values("time").reset_index(drop=True)

    win = np.timedelta64(int(float(sno_window_minutes) * 60), "s")
    times = df["time"].to_numpy(dtype="datetime64[ns]")

    out = []
    for i, t in enumerate(times):
        right = np.searchsorted(times, t + win, side="right")
        for j in range(i + 1, right):
            if df.loc[i, "sat_name"] == df.loc[j, "sat_name"]:
                continue
            dt_min = float((times[j] - t) / np.timedelta64(1, "m"))
            out.append({
                "time_a": df.loc[i, "time"],
                "sat_a": df.loc[i, "sat_name"],
                "scene_a": df.loc[i, "scene_id"],
                "collection_a": df.loc[i, "collection"],

                "time_b": df.loc[j, "time"],
                "sat_b": df.loc[j, "sat_name"],
                "scene_b": df.loc[j, "scene_id"],
                "collection_b": df.loc[j, "collection"],

                "dt_minutes": abs(dt_min),
            })

    return (pd.DataFrame(out).sort_values("dt_minutes").reset_index(drop=True)) if out else pd.DataFrame()


def add_pair_flag_to_sno_table(df_sno: pd.DataFrame, df_events_w: pd.DataFrame) -> pd.DataFrame:
    """
    Adds PAIR_FLAG to each SNO row (GOOD/OK/BAD) based on cloud_scene_pct on either side.
    If either scene is BAD => BAD
    Else if both GOOD => GOOD
    Else => OK
    """
    if df_sno is None or df_sno.empty:
        return pd.DataFrame()

    out = df_sno.copy()

    if df_events_w is None or df_events_w.empty or "cloud_scene_pct" not in df_events_w.columns:
        out["PAIR_FLAG"] = "OK"
        return out

    ev = df_events_w.copy()
    ev["time"] = pd.to_datetime(ev["time"]).dt.floor("S")
    ev = ev[["sat_name", "time", "scene_id", "collection", "cloud_scene_pct"]].copy()

    out["time_a"] = pd.to_datetime(out["time_a"]).dt.floor("S")
    out["time_b"] = pd.to_datetime(out["time_b"]).dt.floor("S")

    out = out.merge(
        ev.rename(columns={"sat_name": "sat_a", "time": "time_a", "scene_id": "scene_a", "collection": "collection_a", "cloud_scene_pct": "cloud_a"}),
        on=["sat_a", "time_a", "scene_a", "collection_a"],
        how="left"
    )
    out = out.merge(
        ev.rename(columns={"sat_name": "sat_b", "time": "time_b", "scene_id": "scene_b", "collection": "collection_b", "cloud_scene_pct": "cloud_b"}),
        on=["sat_b", "time_b", "scene_b", "collection_b"],
        how="left"
    )

    def row_flag(ca, cb) -> str:
        ca_ok = pd.notna(ca)
        cb_ok = pd.notna(cb)
        ca_v = float(ca) if ca_ok else None
        cb_v = float(cb) if cb_ok else None

        if (ca_v is not None and ca_v >= BAD_CLOUD_PCT) or (cb_v is not None and cb_v >= BAD_CLOUD_PCT):
            return "BAD"
        if (ca_v is not None and cb_v is not None and ca_v <= GOOD_CLOUD_PCT and cb_v <= GOOD_CLOUD_PCT):
            return "GOOD"
        return "OK"

    out["PAIR_FLAG"] = [row_flag(a, b) for a, b in zip(out["cloud_a"], out["cloud_b"])]
    # keep table clean
    out = out.drop(columns=["cloud_a", "cloud_b"], errors="ignore")
    return out


# ------------------- FUTURE (TLE) -------------------

def fetch_tle_from_celestrak(norad_id: int) -> str:
    url = f"https://celestrak.org/NORAD/elements/gp.php?CATNR={norad_id}&FORMAT=TLE"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    tle_text = r.text.strip()
    if tle_text.count("\n") < 2:
        raise ValueError(f"Unexpected TLE for NORAD {norad_id}: {tle_text}")
    return tle_text


@st.cache_resource(show_spinner=False)
def get_orbital_cached(sat_name: str) -> Orbital:
    norad = SATELLITE_NORAD.get(sat_name)
    if norad is None:
        raise ValueError(f"No NORAD ID for {sat_name}")
    tle_text = fetch_tle_from_celestrak(norad)
    lines = [l.strip() for l in tle_text.splitlines() if l.strip()]
    name, line1, line2 = lines[0], lines[1], lines[2]
    return Orbital(name, line1=line1, line2=line2)


@st.cache_resource(show_spinner=False)
def get_skyfield_sat_cached(sat_name: str):
    if not SKYFIELD_AVAILABLE:
        raise RuntimeError("Skyfield not installed.")
    norad = SATELLITE_NORAD.get(sat_name)
    if norad is None:
        raise ValueError(f"No NORAD ID for {sat_name}")
    tle_text = fetch_tle_from_celestrak(norad)
    lines = [l.strip() for l in tle_text.splitlines() if l.strip()]
    name, l1, l2 = lines[0], lines[1], lines[2]
    ts = load.timescale()
    sat = EarthSatellite(l1, l2, name, ts)
    return sat, ts


def _default_half_swath_km(sat_name: str) -> Optional[float]:
    s = (sat_name or "").upper()
    if "LANDSAT" in s:
        return 92.5
    if "SENTINEL-2" in s:
        return 145.0
    if "SENTINEL-3" in s:
        return 635.0
    if "MODIS" in s or "TERRA" in s or "AQUA" in s:
        return 1165.0
    if "VIIRS" in s or "NPP" in s:
        return 1520.0
    return None


def _refine_minimum_distance_pyorbital(
    orb: Orbital,
    lat: float,
    lon: float,
    t0: datetime,
    min_elev_deg: float,
    window_s: int = 180,
    step_s: int = 1,
) -> Optional[Tuple[datetime, float, float, float, float, float]]:
    best = None
    obs_alt_m = 0.0
    for dt_s in range(-window_s, window_s + 1, step_s):
        t = t0 + timedelta(seconds=dt_s)
        try:
            sat_lon, sat_lat, alt_km = orb.get_lonlatalt(t)
            az, el = orb.get_observer_look(t, lon, lat, obs_alt_m)
        except Exception:
            continue
        if el <= min_elev_deg:
            continue
        dist_km = great_circle_distance_km(float(sat_lat), float(sat_lon), float(lat), float(lon))
        if best is None or dist_km < best[4]:
            best = (t, float(sat_lat), float(sat_lon), float(alt_km), float(dist_km), float(el))
    return best


@st.cache_data(show_spinner=True)
def predict_future_passes_pyorbital(
    sat_names: Tuple[str, ...],
    lat: float,
    lon: float,
    start_date_str: str,
    end_date_str: str,
    user_tol_km: float,
    min_elev_deg: float,
) -> pd.DataFrame:
    start_time = datetime.strptime(start_date_str + "-00-00-00", "%Y-%m-%d-%H-%M-%S")
    end_time = datetime.strptime(end_date_str + "-23-59-30", "%Y-%m-%d-%H-%M-%S")

    total_days = (end_time - start_time).total_seconds() / 86400.0
    if total_days <= 7:
        step_seconds = 10
    elif total_days <= 31:
        step_seconds = 30
    elif total_days <= 90:
        step_seconds = 60
    else:
        step_seconds = 120

    obs_alt_m = 0.0
    rows: List[Dict[str, Any]] = []

    for sat in sat_names:
        if sat not in SATELLITE_NORAD:
            continue
        try:
            orb = get_orbital_cached(sat)
        except Exception:
            continue

        half_swath = _default_half_swath_km(sat)
        eff_tol = float(user_tol_km) if half_swath is None else max(float(user_tol_km), float(half_swath))

        current = start_time
        in_vis = False
        best_candidate: Optional[Tuple[datetime, float]] = None

        while current <= end_time:
            try:
                sat_lon, sat_lat, alt_km = orb.get_lonlatalt(current)
                az, el = orb.get_observer_look(current, lon, lat, obs_alt_m)
            except Exception:
                current += timedelta(seconds=step_seconds)
                continue

            if el > min_elev_deg:
                dist_km = great_circle_distance_km(float(sat_lat), float(sat_lon), float(lat), float(lon))
                if not in_vis:
                    in_vis = True
                    best_candidate = None
                if best_candidate is None or dist_km < best_candidate[1]:
                    best_candidate = (current, float(dist_km))
            else:
                if in_vis and best_candidate is not None:
                    refined = _refine_minimum_distance_pyorbital(
                        orb, lat, lon, best_candidate[0], min_elev_deg=min_elev_deg, window_s=180, step_s=1
                    )
                    if refined is not None:
                        t, slat, slon, alt, dist, elev = refined
                        if dist <= eff_tol:
                            rows.append({
                                "time": t,
                                "sat_name": sat,
                                "closest_distance_km": dist,
                                "sat_sub_lat": slat,
                                "sat_sub_lon": slon,
                                "alt_km": alt,
                                "elev_deg": elev,
                                "effective_tol_km": eff_tol,
                                "engine": "pyorbital",
                            })
                in_vis = False
                best_candidate = None

            current += timedelta(seconds=step_seconds)

        if in_vis and best_candidate is not None:
            refined = _refine_minimum_distance_pyorbital(
                orb, lat, lon, best_candidate[0], min_elev_deg=min_elev_deg, window_s=180, step_s=1
            )
            if refined is not None:
                t, slat, slon, alt, dist, elev = refined
                if dist <= eff_tol:
                    rows.append({
                        "time": t,
                        "sat_name": sat,
                        "closest_distance_km": dist,
                        "sat_sub_lat": slat,
                        "sat_sub_lon": slon,
                        "alt_km": alt,
                        "elev_deg": elev,
                        "effective_tol_km": eff_tol,
                        "engine": "pyorbital",
                    })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"]).apply(normalize_time_to_sec)
    return df.sort_values(["sat_name", "time"]).reset_index(drop=True)


@st.cache_data(show_spinner=True)
def predict_future_passes_skyfield(
    sat_names: Tuple[str, ...],
    lat: float,
    lon: float,
    start_date_str: str,
    end_date_str: str,
    user_tol_km: float,
    min_elev_deg: float,
) -> pd.DataFrame:
    if not SKYFIELD_AVAILABLE:
        return pd.DataFrame()

    start_dt = datetime.strptime(start_date_str + " 00:00:00", "%Y-%m-%d %H:%M:%S").replace(tzinfo=utc)
    end_dt = datetime.strptime(end_date_str + " 23:59:30", "%Y-%m-%d %H:%M:%S").replace(tzinfo=utc)

    ts = load.timescale()
    t0 = ts.from_datetime(start_dt)
    t1 = ts.from_datetime(end_dt)

    observer = wgs84.latlon(latitude_degrees=lat, longitude_degrees=lon)
    rows: List[Dict[str, Any]] = []

    for sat in sat_names:
        if sat not in SATELLITE_NORAD:
            continue
        try:
            sat_obj, _ = get_skyfield_sat_cached(sat)
        except Exception:
            continue

        half_swath = _default_half_swath_km(sat)
        eff_tol = float(user_tol_km) if half_swath is None else max(float(user_tol_km), float(half_swath))

        try:
            t_events, events = sat_obj.find_events(observer, t0, t1, altitude_degrees=min_elev_deg)
        except Exception:
            continue

        current_triplet = {}
        pass_triplets = []
        for te, ev in zip(t_events, events):
            if ev == 0:
                current_triplet = {"rise": te}
            elif ev == 1:
                current_triplet["culm"] = te
            elif ev == 2:
                current_triplet["set"] = te
                if "culm" in current_triplet:
                    pass_triplets.append(current_triplet)
                current_triplet = {}

        for trip in pass_triplets:
            tc = trip.get("culm")
            if tc is None:
                continue

            minutes = 3
            step_s = 2
            dt_list = np.arange(-minutes * 60, minutes * 60 + 1, step_s)
            best = None

            for dt_s in dt_list:
                tt_dt = tc.utc_datetime().replace(tzinfo=utc) + timedelta(seconds=int(dt_s))
                tt = ts.from_datetime(tt_dt)
                topocentric = (sat_obj - observer).at(tt)
                alt, az, distance = topocentric.altaz()
                elev_deg = float(alt.degrees)
                if elev_deg <= min_elev_deg:
                    continue

                sp = sat_obj.at(tt).subpoint()
                slat = float(sp.latitude.degrees)
                slon = float(sp.longitude.degrees)
                alt_km = float(sp.elevation.km)
                dist_km = great_circle_distance_km(slat, slon, float(lat), float(lon))

                if best is None or dist_km < best["dist_km"]:
                    best = {
                        "time": tt.utc_datetime(),
                        "sat_sub_lat": slat,
                        "sat_sub_lon": slon,
                        "alt_km": alt_km,
                        "dist_km": dist_km,
                        "elev_deg": elev_deg,
                    }

            if best is not None and best["dist_km"] <= eff_tol:
                rows.append({
                    "time": best["time"],
                    "sat_name": sat,
                    "closest_distance_km": best["dist_km"],
                    "sat_sub_lat": best["sat_sub_lat"],
                    "sat_sub_lon": best["sat_sub_lon"],
                    "alt_km": best["alt_km"],
                    "elev_deg": best["elev_deg"],
                    "effective_tol_km": eff_tol,
                    "engine": "skyfield",
                })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"]).apply(normalize_time_to_sec)
    return df.sort_values(["sat_name", "time"]).reset_index(drop=True)


def compute_future_snos(df_pred: pd.DataFrame, sno_window_minutes: float) -> pd.DataFrame:
    if df_pred is None or df_pred.empty:
        return pd.DataFrame()

    df = df_pred.copy()
    df["time"] = pd.to_datetime(df["time"]).apply(normalize_time_to_sec)
    df = df.sort_values("time").reset_index(drop=True)

    times = df["time"].to_numpy(dtype="datetime64[ns]")
    tol = np.timedelta64(int(float(sno_window_minutes) * 60), "s")

    out = []
    for i, t in enumerate(times):
        right = np.searchsorted(times, t + tol, side="right")
        for j in range(i + 1, right):
            if df.loc[i, "sat_name"] == df.loc[j, "sat_name"]:
                continue
            dt_min = float((times[j] - t) / np.timedelta64(1, "m"))
            out.append({
                "sat_a": df.loc[i, "sat_name"],
                "time_a": df.loc[i, "time"],
                "sat_b": df.loc[j, "sat_name"],
                "time_b": df.loc[j, "time"],
                "dt_minutes": abs(dt_min),
            })

    return (pd.DataFrame(out).sort_values("dt_minutes").reset_index(drop=True)) if out else pd.DataFrame()


# ------------------- METRICS -------------------

def acquisition_metrics(df: pd.DataFrame, sat_col: str = "sat_name") -> pd.DataFrame:
    if df is None or df.empty or sat_col not in df.columns:
        return pd.DataFrame()
    tmp = df.copy()
    if "quality_label" not in tmp.columns:
        tmp["quality_label"] = "Unknown"
    g = tmp.groupby(sat_col)["quality_label"].value_counts().unstack(fill_value=0)
    # keep only GOOD/OK/BAD (drop Unknown)
    for col in ["GOOD", "OK", "BAD"]:
        if col not in g.columns:
            g[col] = 0
    g = g[["GOOD", "OK", "BAD"]]
    g["TOTAL"] = g.sum(axis=1)
    g = g.reset_index().rename(columns={sat_col: "sat_name"})
    g = g[["sat_name", "TOTAL", "GOOD", "OK", "BAD"]]
    return g.sort_values(["TOTAL", "sat_name"], ascending=[False, True]).reset_index(drop=True)


def sno_metrics_by_pair_counts(df_sno: pd.DataFrame) -> pd.DataFrame:
    """
    Expects df_sno has columns: sat_a, sat_b, dt_minutes, PAIR_FLAG (GOOD/OK/BAD)
    Output: pair, SNO_COUNT, GOOD_COUNT, OK_COUNT, BAD_COUNT, SNO_WINDOW(min/hours/days)
    """
    if df_sno is None or df_sno.empty:
        return pd.DataFrame()
    tmp = df_sno.copy()
    tmp["pair"] = tmp["sat_a"].astype(str) + " ↔ " + tmp["sat_b"].astype(str)
    if "PAIR_FLAG" not in tmp.columns:
        tmp["PAIR_FLAG"] = "OK"

    # counts by flag
    counts = tmp.pivot_table(index="pair", columns="PAIR_FLAG", values="dt_minutes", aggfunc="size", fill_value=0)
    for col in ["GOOD", "OK", "BAD"]:
        if col not in counts.columns:
            counts[col] = 0
    counts = counts[["GOOD", "OK", "BAD"]]
    counts = counts.rename(columns={"GOOD": "GOOD_COUNT", "OK": "OK_COUNT", "BAD": "BAD_COUNT"})

    # total + min dt
    agg = tmp.groupby("pair").agg(SNO_COUNT=("pair", "size"), DT_MIN=("dt_minutes", "min")).reset_index()

    out = agg.merge(counts.reset_index(), on="pair", how="left")
    out["SNO_WINDOW(min/hours/days)"] = out["DT_MIN"].apply(format_dt_minutes)
    out = out.drop(columns=["DT_MIN"])
    out = out.sort_values(["SNO_COUNT", "pair"], ascending=[False, True]).reset_index(drop=True)
    return out


# ------------------- TIME SERIES -------------------

def plot_timeseries(df: pd.DataFrame, ycols: List[str], title: str):
    if df is None or df.empty:
        st.info("No data to plot.")
        return
    df = df.sort_values("time").copy()
    fig, ax = plt.subplots(figsize=(10, 3))
    plotted = False
    for c in ycols:
        if c in df.columns and df[c].notna().any():
            ax.plot(df["time"], df[c], marker="o", linestyle="-", label=c)
            plotted = True
    if not plotted:
        st.info("No reflectance columns available (enable 'Also sample reflectance').")
        return
    ax.set_title(title)
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Reflectance")
    ax.grid(True)
    ax.legend()
    fig.autofmt_xdate()
    st.pyplot(fig)


# ------------------- MAIN APP -------------------

def main():

    # Canonical state
    if "lat" not in st.session_state:
        st.session_state["lat"] = float(DEFAULT_LAT)
    if "lon" not in st.session_state:
        st.session_state["lon"] = float(DEFAULT_LON)
    if "_last_click" not in st.session_state:
        st.session_state["_last_click"] = None
    if "mode" not in st.session_state:
        st.session_state["mode"] = "Past acquisitions"
    if "date_range" not in st.session_state:
        today = datetime.utcnow().date()
        st.session_state["date_range"] = (today - timedelta(days=90), today - timedelta(days=1))
    if "site_choice" not in st.session_state:
        st.session_state["site_choice"] = "Crater Lake (OR)"
    if "results_dirty" not in st.session_state:
        st.session_state["results_dirty"] = False
    if "compute_requested" not in st.session_state:
        st.session_state["compute_requested"] = False
    if "dirty_reason" not in st.session_state:
        st.session_state["dirty_reason"] = None

    # Apply map updates before widgets
    if "map_lat" in st.session_state and "map_lon" in st.session_state:
        st.session_state["lat"] = float(st.session_state.pop("map_lat"))
        st.session_state["lon"] = float(st.session_state.pop("map_lon"))

    st.session_state["lat_input"] = float(st.session_state["lat"])
    st.session_state["lon_input"] = float(st.session_state["lon"])

    # Persist map view
    if "map_view_center" not in st.session_state:
        st.session_state["map_view_center"] = [float(st.session_state["lat"]), float(st.session_state["lon"])]
    if "map_view_zoom" not in st.session_state:
        st.session_state["map_view_zoom"] = 7

    # GEE init
    try:
        mode_gee = init_ee(st.session_state["submitted_project_id"])
        st.caption(f"Earth Engine initialized using: {mode_gee}")
    except Exception as e:
        st.error(str(e))
        st.stop()

    # Sidebar
    st.sidebar.header("Site & Period")

    site_choice = st.sidebar.selectbox(
        "Quick test sites",
        options=list(TEST_SITES.keys()),
        key="site_choice",
        on_change=mark_results_dirty,
    )
    if site_choice != st.session_state.get("_site_choice_applied"):
        st.session_state["_site_choice_applied"] = site_choice
        if site_choice != "Custom (use inputs)" and TEST_SITES[site_choice] is not None:
            lat0, lon0 = TEST_SITES[site_choice]
            st.session_state["lat"] = float(lat0)
            st.session_state["lon"] = float(lon0)
            st.session_state["lat_input"] = float(lat0)
            st.session_state["lon_input"] = float(lon0)
            st.session_state["map_view_center"] = [float(lat0), float(lon0)]
            st.session_state["map_view_zoom"] = 7
            mark_results_dirty()

    today = datetime.utcnow().date()
    tomorrow = today + timedelta(days=1)
    yesterday = today - timedelta(days=1)

    mode_choice = st.sidebar.radio(
        "Choose",
        options=["Past acquisitions", "Future pass planning"],
        index=0 if st.session_state["mode"] == "Past acquisitions" else 1,
        key="mode_choice",
        on_change=mark_results_dirty,
    )
    if mode_choice != st.session_state["mode"]:
        st.session_state["mode"] = mode_choice
        if mode_choice == "Future pass planning":
            st.session_state["date_range"] = (tomorrow, tomorrow + timedelta(days=30))
        else:
            st.session_state["date_range"] = (today - timedelta(days=90), yesterday)
        mark_results_dirty()
        st.rerun()

    def on_latlon_change():
        st.session_state["lat"] = float(st.session_state["lat_input"])
        st.session_state["lon"] = float(st.session_state["lon_input"])
        st.session_state["map_view_center"] = [float(st.session_state["lat"]), float(st.session_state["lon"])]
        mark_results_dirty()

    def request_compute() -> None:
        st.session_state["compute_requested"] = True

    st.sidebar.number_input(
        "Latitude (deg)",
        min_value=-90.0, max_value=90.0,
        step=0.0001, format="%.6f",
        key="lat_input",
        on_change=on_latlon_change,
    )
    st.sidebar.number_input(
        "Longitude (deg)",
        min_value=-180.0, max_value=180.0,
        step=0.0001, format="%.6f",
        key="lon_input",
        on_change=on_latlon_change,
    )

    # Date constraints
    if st.session_state["mode"] == "Future pass planning":
        min_d = tomorrow
        max_d = tomorrow + timedelta(days=365 * 3)
    else:
        min_d = date(1972, 1, 1)
        max_d = yesterday

    date_range = st.sidebar.date_input(
        "Date range",
        value=st.session_state["date_range"],
        min_value=min_d,
        max_value=max_d,
        key="date_range_widget",
        on_change=mark_results_dirty,
    )

    if isinstance(date_range, tuple):
        start_date, end_date = date_range
    else:
        start_date = date_range
        end_date = date_range

    if start_date > end_date:
        start_date, end_date = end_date, start_date

    if st.session_state["mode"] == "Future pass planning":
        if start_date < tomorrow:
            start_date = tomorrow
        if end_date < start_date:
            end_date = start_date
    else:
        if end_date > yesterday:
            end_date = yesterday
        if end_date < start_date:
            start_date = end_date

    st.session_state["date_range"] = (start_date, end_date)
    start_date_str = start_date.isoformat()
    end_date_str = end_date.isoformat()

    st.sidebar.write("### Missions / Satellites")

    if st.session_state["mode"] == "Past acquisitions":
        selected_past = st.sidebar.multiselect(
            "Select missions (past acquisitions in GEE)",
            options=list(PAST_MISSIONS.keys()),
            default=["LANDSAT 8", "LANDSAT 9", "SENTINEL-2A", "SENTINEL-2B"],
            key="selected_past",
            on_change=mark_results_dirty,
        )
        selected_future = []
    else:
        selected_future = st.sidebar.multiselect(
            "Select satellites (future via TLE)",
            options=list(SATELLITE_NORAD.keys()),
            default=["LANDSAT 8", "LANDSAT 9", "SENTINEL-2A", "SENTINEL-2B"],
            key="selected_future",
            on_change=mark_results_dirty,
        )
        selected_past = []

    st.sidebar.write("### Settings")

    # SNO window selector (minutes to multi-day)
    SNO_PRESETS = [
        ("30 min", 30),
        ("1 hour", 60),
        ("2 hours", 120),
        ("3 hours", 180),
        ("4 hours", 240),
        ("6 hours", 360),
        ("8 hours", 480),
        ("12 hours", 720),
        ("18 hours", 1080),
        ("24 hours", 1440),
        ("1 day", 1440),
        ("2 days", 2 * 1440),
        ("3 days", 3 * 1440),
        ("4 days", 4 * 1440),
        ("5 days", 5 * 1440),
        ("10 days", 10 * 1440),
        ("Custom…", None),
    ]
    preset_labels = [x[0] for x in SNO_PRESETS]
    default_label = "30 min"

    sno_choice = st.sidebar.selectbox(
        "SNO time window",
        options=preset_labels,
        index=preset_labels.index(default_label),
        help="Short windows for modern constellations; longer windows for early Landsat era.",
        key="sno_window_choice",
        on_change=mark_results_dirty,
    )
    sno_window_min = dict(SNO_PRESETS).get(sno_choice)
    if sno_window_min is None:
        sno_window_min = st.sidebar.number_input(
            "Custom SNO window (minutes)",
            min_value=30,
            max_value=10 * 1440,
            value=60,
            step=30,
            help="Enter minutes (up to 10 days).",
            key="sno_window_custom",
            on_change=mark_results_dirty,
        )
    sno_window_min = float(sno_window_min)
    st.session_state["sno_window_min"] = sno_window_min

    site_buffer_m = st.sidebar.slider(
        "Site buffer radius for past acquisitions (m)",
        min_value=10, max_value=2000,
        value=int(DEFAULT_SITE_BUFFER_M),
        step=10,
        help="Past acquisitions counted if image footprint intersects this buffer around the point.",
        key="site_buffer_m",
        on_change=mark_results_dirty,
    )

    sample_reflectance = st.sidebar.checkbox(
        "Also sample reflectance at site for MODIS/VIIRS only (slower)",
        value=False,
        help="Landsat and Sentinel-2 TOA reflectance sampling has been removed.",
        key="sample_reflectance",
        on_change=mark_results_dirty,
    )

    if st.session_state["mode"] == "Future pass planning":
        overpass_tol_km = st.sidebar.slider(
            "Future pass tolerance to point (km)",
            min_value=1.0, max_value=2000.0,
            value=float(DEFAULT_OVERPASS_TOL_KM),
            step=1.0,
            key="overpass_tol_km",
            on_change=mark_results_dirty,
        )
        min_elev_deg = st.sidebar.slider(
            "Minimum elevation (deg)",
            min_value=0.0, max_value=30.0,
            value=float(MIN_ELEV_DEG_DEFAULT),
            step=1.0,
            key="min_elev_deg",
            on_change=mark_results_dirty,
        )
        future_engine = st.sidebar.radio(
            "Future engine",
            options=[
                "Skyfield (more accurate)" if SKYFIELD_AVAILABLE else "Skyfield (install required)",
                "Pyorbital (fallback)"
            ],
            index=0 if SKYFIELD_AVAILABLE else 1,
            help="Install Skyfield: pip install skyfield sgp4",
            key="future_engine",
            on_change=mark_results_dirty,
        )
    else:
        overpass_tol_km = float(DEFAULT_OVERPASS_TOL_KM)
        min_elev_deg = float(MIN_ELEV_DEG_DEFAULT)
        future_engine = "Pyorbital (fallback)"

    st.sidebar.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
    st.sidebar.markdown(
        "<div style='font-size:0.95rem; font-weight:700; color:#d9480f; margin-top:8px;'>Reset everything</div>",
        unsafe_allow_html=True,
    )
    if st.sidebar.button("Reset App", type="primary", use_container_width=True):
        reset_app_state()
        st.rerun()

    current_compute_params = {
        "mode": st.session_state["mode"],
        "lat": round(float(st.session_state["lat"]), 6),
        "lon": round(float(st.session_state["lon"]), 6),
        "start_date": start_date_str,
        "end_date": end_date_str,
        "selected_past": tuple(selected_past),
        "selected_future": tuple(selected_future),
        "sno_window_min": float(sno_window_min),
        "site_buffer_m": float(site_buffer_m),
        "sample_reflectance": bool(sample_reflectance),
        "overpass_tol_km": float(overpass_tol_km),
        "min_elev_deg": float(min_elev_deg),
        "future_engine": future_engine,
    }

    # -------------------- MAP (TOP) --------------------
    st.subheader("Select site on map (results overlay appears here after Compute)")
    st.caption("Click empty map area to set the site. Zoom and pan should stay where you leave them.")

    lat = float(st.session_state["lat"])
    lon = float(st.session_state["lon"])

    center = st.session_state.get("map_view_center", [lat, lon])
    zoom = int(st.session_state.get("map_view_zoom", 7))

    m = folium.Map(location=center, zoom_start=zoom, control_scale=True, tiles=None)
    folium.TileLayer("CartoDB positron", name="Light", control=True).add_to(m)
    folium.TileLayer("OpenStreetMap", name="OSM", control=True).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Satellite (Esri)",
        overlay=False,
        control=True
    ).add_to(m)

    # selected site marker always at actual site
    folium.Marker(
        location=[lat, lon],
        tooltip="Selected location",
        icon=folium.Icon(color="red", icon="map-marker", prefix="fa"),
    ).add_to(m)

    MousePosition(position="topright", separator=" | ", prefix="Lat/Lon:", num_digits=6).add_to(m)
    folium.LatLngPopup().add_to(m)

    Draw(
        export=False,
        draw_options={
            "polyline": False,
            "polygon": False,
            "circle": False,
            "circlemarker": False,
            "rectangle": False,
            "marker": True,
        },
        edit_options={"edit": True, "remove": True},
    ).add_to(m)

    # ----- runtime overlay on map (optional, shown when available) -----
    if "runtime_s" in st.session_state:
        rt_txt = format_runtime(float(st.session_state["runtime_s"]))
        runtime_html = f"""
        <div style="position: fixed; top: 12px; right: 14px; z-index: 9999;
             background: rgba(11, 19, 32, 0.92);
             padding: 10px 12px; border-radius: 12px;
             border: 1px solid #22304a;">
          <span style="font-size: 18px; font-weight: 900; color: #ff3333;">
            Runtime: {rt_txt}
          </span>
        </div>
        """
        m.get_root().html.add_child(folium.Element(runtime_html))

    # ---- Overlay Past Results ----
    if (not st.session_state.get("results_dirty")) and st.session_state.get("mode") == "Past acquisitions" and "past_df_raw" in st.session_state:
        df_raw = st.session_state.get("past_df_raw", pd.DataFrame())
        df_sno = st.session_state.get("past_sno", pd.DataFrame())
        selected = st.session_state.get("past_selected", [])

        star_cluster = MarkerCluster(
            name="Acquisitions ★",
            options={
                "spiderfyOnMaxZoom": True,
                "showCoverageOnHover": False,
                "zoomToBoundsOnClick": True,
                "disableClusteringAtZoom": 12,
            },
        ).add_to(m)

        if df_raw is not None and not df_raw.empty:
            df_plot = df_raw.copy()
            df_plot["time"] = pd.to_datetime(df_plot["time"]).apply(normalize_time_to_sec)
            for c in ["scene_center_lat", "scene_center_lon"]:
                df_plot[c] = pd.to_numeric(df_plot[c], errors="coerce")

            df_idx = df_plot.dropna(subset=["scene_center_lat", "scene_center_lon"]).copy()
            df_idx = df_idx.set_index(["sat_name", "time", "scene_id", "collection"], drop=False)

            for _, r in df_plot.dropna(subset=["scene_center_lat", "scene_center_lon"]).iterrows():
                mission = str(r.get("sat_name", ""))
                hexcol = mission_hex_color(mission)
                popup = (
                    f"<b>{mission}</b><br>"
                    f"time={r.get('time')}<br>"
                    f"scene={r.get('scene_id')}<br>"
                    f"collection={r.get('collection','')}"
                )
                add_star(
                    star_cluster,
                    float(r["scene_center_lat"]),
                    float(r["scene_center_lon"]),
                    color_hex=hexcol,
                    tooltip=mission,
                    popup_html=popup
                )

            if df_sno is not None and not df_sno.empty:
                sno_layer = folium.FeatureGroup(name="SNO rings", show=True).add_to(m)
                for _, srow in df_sno.iterrows():
                    try:
                        kA = (
                            str(srow["sat_a"]),
                            normalize_time_to_sec(srow["time_a"]),
                            srow["scene_a"],
                            srow["collection_a"],
                        )
                        kB = (
                            str(srow["sat_b"]),
                            normalize_time_to_sec(srow["time_b"]),
                            srow["scene_b"],
                            srow["collection_b"],
                        )
                        if kA in df_idx.index:
                            rrA = df_idx.loc[kA]
                            add_sno_ring(sno_layer, float(rrA["scene_center_lat"]), float(rrA["scene_center_lon"]))
                        if kB in df_idx.index:
                            rrB = df_idx.loc[kB]
                            add_sno_ring(sno_layer, float(rrB["scene_center_lat"]), float(rrB["scene_center_lon"]))
                    except Exception:
                        continue


    # ---- Overlay Future Results ----
    if (not st.session_state.get("results_dirty")) and st.session_state.get("mode") == "Future pass planning" and "future_df_raw" in st.session_state:
        df_raw = st.session_state.get("future_df_raw", pd.DataFrame())
        df_sno = st.session_state.get("future_sno", pd.DataFrame())
        selected = st.session_state.get("future_selected", [])

        star_cluster = MarkerCluster(name="Passes ★").add_to(m)

        if df_raw is not None and not df_raw.empty:
            df_plot = df_raw.copy()
            df_plot["time"] = pd.to_datetime(df_plot["time"]).apply(normalize_time_to_sec)
            for c in ["sat_sub_lat", "sat_sub_lon"]:
                df_plot[c] = pd.to_numeric(df_plot[c], errors="coerce")

            df_idx = df_plot.dropna(subset=["sat_sub_lat", "sat_sub_lon"]).copy()
            df_idx = df_idx.set_index(["sat_name", "time"], drop=False)

            for _, r in df_plot.dropna(subset=["sat_sub_lat", "sat_sub_lon"]).iterrows():
                sat = str(r.get("sat_name", ""))
                hexcol = mission_hex_color(sat)
                popup = (
                    f"<b>{sat}</b><br>"
                    f"time={r.get('time')}<br>"
                    f"dist={float(r.get('closest_distance_km', np.nan)):.1f} km<br>"
                    f"elev={float(r.get('elev_deg', np.nan)):.1f}°<br>"
                    f"engine={r.get('engine','')}"
                )
                add_star(
                    star_cluster,
                    float(r["sat_sub_lat"]),
                    float(r["sat_sub_lon"]),
                    color_hex=hexcol,
                    tooltip=sat,
                    popup_html=popup
                )

            if df_sno is not None and not df_sno.empty:
                sno_layer = folium.FeatureGroup(name="SNO rings", show=True).add_to(m)
                for _, srow in df_sno.iterrows():
                    try:
                        kA = (str(srow["sat_a"]), normalize_time_to_sec(srow["time_a"]))
                        kB = (str(srow["sat_b"]), normalize_time_to_sec(srow["time_b"]))
                        if kA in df_idx.index:
                            rrA = df_idx.loc[kA]
                            add_sno_ring(sno_layer, float(rrA["sat_sub_lat"]), float(rrA["sat_sub_lon"]))
                        if kB in df_idx.index:
                            rrB = df_idx.loc[kB]
                            add_sno_ring(sno_layer, float(rrB["sat_sub_lat"]), float(rrB["sat_sub_lon"]))
                    except Exception:
                        continue


    folium.LayerControl().add_to(m)
     
    map_data = st_folium(
    m,
    height=450,
    width="stretch",
    key="site_map",
    returned_objects=["last_clicked", "last_object_clicked", "center", "zoom", "last_active_drawing", "last_drawn", "all_drawings"],
)

    if map_data:
       if map_data.get("center"):
           st.session_state["map_view_center"] = [
            float(map_data["center"]["lat"]),
            float(map_data["center"]["lng"]),]
    if map_data.get("zoom") is not None:
        st.session_state["map_view_zoom"] = int(map_data["zoom"])


    def request_map_update(new_lat: float, new_lon: float):
    	this_click = (round(new_lat, 6), round(new_lon, 6))
    	if st.session_state.get("_last_click") != this_click:
           st.session_state["_last_click"] = this_click
           st.session_state["map_lat"] = float(new_lat)
           st.session_state["map_lon"] = float(new_lon)
           st.session_state["map_view_center"] = [float(new_lat), float(new_lon)]
           st.session_state["results_dirty"] = True
           st.session_state["dirty_reason"] = "location"
           #st.rerun() 


     # Only background click will update site
    if map_data and map_data.get("last_clicked") and not map_data.get("last_object_clicked"):
        request_map_update(
        float(map_data["last_clicked"]["lat"]),
        float(map_data["last_clicked"]["lng"]),
    )
   
    # Draw/edit marker-to-set
    candidate = None
    if map_data:
        candidate = map_data.get("last_active_drawing") or map_data.get("last_drawn")
        if (candidate is None) and map_data.get("all_drawings"):
            candidate = map_data["all_drawings"][-1]
    if candidate:
        geom = candidate.get("geometry", {})
        if geom.get("type") == "Point":
            coords = geom.get("coordinates", None)  # [lon, lat]
            if coords and len(coords) == 2:
                request_map_update(float(coords[1]), float(coords[0]))

    st.markdown(f"**Selected location:** {lat:.6f}, {lon:.6f}")

    # -------------------- Tabs AFTER map --------------------
    tab_about, tab_over, tab_metrics = st.tabs(
        ["About", "Overpasses & Weather", "Metrics"]
    )

    with tab_about:
        st.markdown(ABOUT_MD)
        if not SKYFIELD_AVAILABLE:
            st.info("Skyfield not installed. To enable: `pip install skyfield sgp4`")

    with tab_over:
        st.subheader("Compute results")
        if st.session_state.get("results_dirty"):
            st.info("")

        with st.form("compute_form"):
            submitted = st.form_submit_button("Compute", type="primary")

        if submitted:
            t_start = time.perf_counter()

            # save a start time so runtime exists even if prior state missing
            st.session_state["run_start_time"] = time.time()

            if st.session_state["mode"] == "Past acquisitions":
                if not selected_past:
                    st.warning("Select at least one past mission.")
                else:
                    with st.spinner("Fetching past acquisitions from GEE (accurate)..."):
                        df_events = gee_past_acquisitions(
                            lat=lat, lon=lon,
                            start_date=start_date_str, end_date=end_date_str,
                            selected_missions=tuple(selected_past),
                            site_buffer_m=float(site_buffer_m),
                            sample_reflectance=bool(sample_reflectance),
                        )

                    if df_events.empty:
                        st.session_state["past_df_raw"] = df_events
                        st.session_state["past_df"] = pd.DataFrame()
                        st.session_state["past_sno"] = pd.DataFrame()
                        st.warning(
                            "No acquisitions found. "
                            "If MSS, use 1970s–1990s. If MODIS/VIIRS, use 2000+."
                        )
                    else:
                        with st.spinner("Fetching weather and attaching..."):
                            df_hourly = fetch_hourly_weather(lat, lon, start_date, end_date)
                            df_events_w = attach_weather(df_events, df_hourly)

                        df_sno = compute_snos_allpairs(df_events, float(sno_window_min))
                        # add per-row PAIR_FLAG so metrics can count GOOD/OK/BAD from SNO table directly
                        df_sno = add_pair_flag_to_sno_table(df_sno, df_events_w)

                        st.session_state["past_df_raw"] = df_events
                        st.session_state["past_df"] = df_events_w
                        st.session_state["past_sno"] = df_sno
                        st.session_state["past_selected"] = list(selected_past)

            else:
                if not selected_future:
                    st.warning("Select at least one satellite for future planning.")
                else:
                    with st.spinner("Predicting future passes..."):
                        use_skyfield = SKYFIELD_AVAILABLE and ("Skyfield" in future_engine)
                        if use_skyfield:
                            df_pred = predict_future_passes_skyfield(
                                sat_names=tuple(selected_future),
                                lat=lat, lon=lon,
                                start_date_str=start_date_str, end_date_str=end_date_str,
                                user_tol_km=float(overpass_tol_km),
                                min_elev_deg=float(min_elev_deg),
                            )
                        else:
                            df_pred = predict_future_passes_pyorbital(
                                sat_names=tuple(selected_future),
                                lat=lat, lon=lon,
                                start_date_str=start_date_str, end_date_str=end_date_str,
                                user_tol_km=float(overpass_tol_km),
                                min_elev_deg=float(min_elev_deg),
                            )

                    if df_pred.empty:
                        st.session_state["future_df_raw"] = df_pred
                        st.session_state["future_df"] = pd.DataFrame()
                        st.session_state["future_sno"] = pd.DataFrame()
                        st.warning("No visible passes found within tolerance. Try increasing tolerance/date window.")
                    else:
                        with st.spinner("Fetching weather and attaching..."):
                            df_hourly = fetch_hourly_weather(lat, lon, start_date, end_date)
                            df_pred_w = attach_weather(df_pred, df_hourly)

                        df_sno_f = compute_future_snos(df_pred, float(sno_window_min))
                        # future SNOs don't have scene cloud; use weather-based label if you want
                        # For now keep simple: OK
                        if not df_sno_f.empty:
                            df_sno_f["PAIR_FLAG"] = "OK"

                        st.session_state["future_df_raw"] = df_pred
                        st.session_state["future_df"] = df_pred_w
                        st.session_state["future_sno"] = df_sno_f
                        st.session_state["future_selected"] = list(selected_future)

            t_end = time.perf_counter()
            st.session_state["runtime_s"] = float(t_end - t_start)
            st.session_state["last_compute_params"] = current_compute_params
            st.session_state["results_dirty"] = False
            st.session_state["dirty_reason"] = None
            st.rerun()

        # Tables
        if st.session_state["mode"] == "Past acquisitions" and "past_df" in st.session_state:
            df_events_w = st.session_state.get("past_df", pd.DataFrame())
            if df_events_w is None or df_events_w.empty:
                st.info("No past acquisitions to display (yet).")
            else:
                st.subheader("Past acquisitions + weather")
                base_cols = [
                    "sat_name", "time", "scene_id", "collection",
                    "cloud_scene_pct",
                    "cloud_cover_pct", "precip_mm", "quality_label",
                    "scene_center_dist_km", "scene_center_lat", "scene_center_lon",
                ]
                refl_cols = [c for c in df_events_w.columns if c.startswith("refl_")]
                cols = base_cols + refl_cols
                for c in cols:
                    if c not in df_events_w.columns:
                        df_events_w[c] = np.nan
                df_show = df_events_w[cols].sort_values(["time", "sat_name"])
                st.dataframe(style_quality_rows(df_show), use_container_width=True)

                st.subheader("SNO candidates (past)")
                df_sno = st.session_state.get("past_sno", pd.DataFrame())
                if df_sno is None or df_sno.empty:
                    st.info("No SNO candidates found for this window.")
                else:
                    # show PAIR_FLAG as user wanted (no cloud_a/cloud_b)
                    show_cols = [
                        "time_a", "sat_a", "scene_a", "collection_a",
                        "time_b", "sat_b", "scene_b", "collection_b",
                        "dt_minutes", "PAIR_FLAG"
                    ]
                    for c in show_cols:
                        if c not in df_sno.columns:
                            df_sno[c] = np.nan
                    st.dataframe(df_sno[show_cols].sort_values("dt_minutes"), use_container_width=True)

        if st.session_state["mode"] == "Future pass planning" and "future_df" in st.session_state:
            df_pred_w = st.session_state.get("future_df", pd.DataFrame())
            if df_pred_w is None or df_pred_w.empty:
                st.info("No future passes to display (yet).")
            else:
                st.subheader("Future passes + weather")
                cols = [
                    "sat_name", "time", "engine",
                    "cloud_cover_pct", "precip_mm", "quality_label",
                    "closest_distance_km", "elev_deg", "alt_km",
                    "sat_sub_lat", "sat_sub_lon",
                ]
                for c in cols:
                    if c not in df_pred_w.columns:
                        df_pred_w[c] = np.nan
                df_show = df_pred_w[cols].sort_values(["time", "sat_name"])
                st.dataframe(style_quality_rows(df_show), use_container_width=True)

                st.subheader("SNO candidates (future)")
                df_sno_f = st.session_state.get("future_sno", pd.DataFrame())
                if df_sno_f is None or df_sno_f.empty:
                    st.info("No SNO candidates found for this window.")
                else:
                    st.dataframe(df_sno_f.sort_values("dt_minutes"), use_container_width=True)

    with tab_metrics:
        st.subheader("Metrics (per sensor and per pair)")

        # Site context
        site_label = st.session_state.get("site_choice", "Custom")
        st.markdown(
            f"**Site:** {site_label} &nbsp;&nbsp; "
            f"**Lat/Lon:** {lat:.4f}, {lon:.4f} &nbsp;&nbsp; "
            f"**Dates:** {start_date_str} to {end_date_str} &nbsp;&nbsp; "
            f"**SNO window:** {format_dt_minutes(sno_window_min)}"
        )

        if st.session_state["mode"] == "Past acquisitions":
            df_events_w = st.session_state.get("past_df", pd.DataFrame())
            df_sno = st.session_state.get("past_sno", pd.DataFrame())

            st.markdown("### Acquisition quality counts (by sensor)")
            m1 = acquisition_metrics(df_events_w, sat_col="sat_name")
            if m1.empty:
                st.info("Run Past acquisitions first.")
            else:
                st.dataframe(m1, use_container_width=True)

            st.markdown("### SNO metrics (by sensor pair)")
            if df_sno is None or df_sno.empty:
                st.info("No SNOs found (or run Past acquisitions first).")
            else:
                m2 = sno_metrics_by_pair_counts(df_sno)
                st.dataframe(m2, use_container_width=True)

        else:
            df_pred_w = st.session_state.get("future_df", pd.DataFrame())
            df_sno_f = st.session_state.get("future_sno", pd.DataFrame())

            st.markdown("### Pass quality counts (by sensor)")
            m1 = acquisition_metrics(df_pred_w, sat_col="sat_name")
            if m1.empty:
                st.info("Run Future pass planning first.")
            else:
                st.dataframe(m1, use_container_width=True)

            st.markdown("### SNO metrics (future) (by sensor pair)")
            if df_sno_f is None or df_sno_f.empty:
                st.info("No SNOs found (or run Future pass planning first).")
            else:
                # future df_sno_f has PAIR_FLAG maybe OK; still show counts
                if "PAIR_FLAG" not in df_sno_f.columns:
                    df_sno_f = df_sno_f.copy()
                    df_sno_f["PAIR_FLAG"] = "OK"
                m2 = sno_metrics_by_pair_counts(df_sno_f.rename(columns={"sat_name": "sat_a"}))  # harmless
                # For future table, the pair columns are sat_a/sat_b already; keep as is:
                m2 = sno_metrics_by_pair_counts(df_sno_f)
                st.dataframe(m2, use_container_width=True)



if __name__ == "__main__":
    main()



