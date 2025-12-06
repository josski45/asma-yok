# dashboard.py - ESP32 IoT Dashboard
import streamlit as st
import pandas as pd
import numpy as np
import json
import time
import queue
import threading
import joblib
from datetime import datetime, timezone, timedelta
import paho.mqtt.client as mqtt
from streamlit_autorefresh import st_autorefresh

# ==================== KONFIGURASI ====================
MQTT_BROKER = "broker.emqx.io"
MQTT_PORT = 1883
TOPIC_DATA = "projek/asma/data_sensor"
TOPIC_LED = "projek/asma/kontrol_led"
TOPIC_PREDICTION = "projek/asma/prediction"
MAX_POINTS = 100
MODEL_PATH = "svm_rbf.pkl"

# ==================== LOAD ML MODEL ====================
@st.cache_resource
def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        return model, None
    except Exception as e:
        error_msg = str(e)
        if "incompatible dtype" in error_msg or "node array" in error_msg:
            return None, "⚠️ Model incompatible! Install scikit-learn==1.1.3: pip install scikit-learn==1.1.3"
        return None, f"❌ Gagal load model: {error_msg}"

model, model_error = load_model()

# ==================== SESSION STATE ====================
if "mqtt_in_q" not in st.session_state:
    st.session_state.mqtt_in_q = queue.Queue()
if "logs" not in st.session_state:
    st.session_state.logs = []
if "last" not in st.session_state:
    st.session_state.last = None
if "mqtt_worker_started" not in st.session_state:
    st.session_state.mqtt_worker_started = False

# ==================== MQTT WORKER ====================
def mqtt_worker(broker, port, topic_sensor, in_q):
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    
    def _on_connect(c, userdata, flags, rc, properties=None):
        if rc == 0:
            c.subscribe(topic_sensor)
    
    def _on_message(c, userdata, msg, properties=None):
        try:
            data = json.loads(msg.payload.decode())
            wib = timezone(timedelta(hours=7))
            in_q.put({
                "ts": datetime.now(wib).strftime("%H:%M:%S"),
                "payload": data
            })
        except:
            pass

    client.on_connect = _on_connect
    client.on_message = _on_message

    while True:
        try:
            client.connect(broker, port, keepalive=60)
            client.loop_forever()
        except:
            time.sleep(2)

# ==================== MQTT PUBLISHER ====================
@st.cache_resource
def get_mqtt_publisher():
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
        client.loop_start()
    except:
        pass
    return client

def send_led(cmd):
    try:
        get_mqtt_publisher().publish(TOPIC_LED, cmd, qos=1)
        return True
    except:
        return False

def publish_prediction(pred_data):
    """Publish hasil prediksi ML ke MQTT"""
    try:
        payload = json.dumps(pred_data)
        get_mqtt_publisher().publish(TOPIC_PREDICTION, payload, qos=1)
        return True
    except:
        return False

# Start worker
if not st.session_state.mqtt_worker_started:
    threading.Thread(target=mqtt_worker, args=(MQTT_BROKER, MQTT_PORT, TOPIC_DATA, st.session_state.mqtt_in_q), daemon=True).start()
    st.session_state.mqtt_worker_started = True

# ==================== PROCESS DATA ====================
def process_incoming():
    q = st.session_state.mqtt_in_q
    while not q.empty():
        item = q.get()
        p = item["payload"]
        
        suhu = p.get("suhu", 0)
        lembab = p.get("lembab", 0)
        gas = p.get("gas", 0)
        
        # ML Prediction
        pred_label = None
        pred_text = "N/A"
        confidence = None
        
        if model is not None:
            try:
                X = [[suhu, lembab, gas]]
                pred_label = model.predict(X)[0]
                
                # Map label ke text
                label_map = {0: "Aman", 1: "Waspada", 2: "Bahaya"}
                pred_text = label_map.get(pred_label, "Unknown")
                
                # Confidence jika tersedia
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)[0]
                    confidence = float(np.max(proba)) * 100
                    
                    # Publish prediction ke MQTT
                    prediction_payload = {
                        "timestamp": datetime.now(timezone(timedelta(hours=7))).isoformat(),
                        "sensor_data": {
                            "temperature": float(suhu),
                            "humidity": float(lembab),
                            "gas_raw": int(gas)
                        },
                        "prediction": {
                            "status": pred_text.upper(),
                            "class": int(pred_label),
                            "confidence": round(confidence, 2),
                            "probabilities": {
                                "aman": round(float(proba[0]) * 100, 2),
                                "waspada": round(float(proba[1]) * 100, 2),
                                "bahaya": round(float(proba[2]) * 100, 2)
                            }
                        }
                    }
                    publish_prediction(prediction_payload)
            except Exception as e:
                pred_text = "Error"
        
        row = {
            "ts": item["ts"],
            "suhu": suhu,
            "lembab": lembab,
            "gas": gas,
            "status": pred_text,
            "confidence": confidence
        }
        
        st.session_state.last = row
        st.session_state.logs.append(row)
        if len(st.session_state.logs) > 5000:
            st.session_state.logs = st.session_state.logs[-5000:]

process_incoming()

# ==================== PAGE CONFIG ====================
st.set_page_config(page_title="ESP32 Dashboard", page_icon="📊", layout="wide")

# Tampilkan warning jika model error
if model_error:
    st.error(model_error, icon="⚠️")

# ==================== CLEAN CSS ====================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .stApp {
        background: #EFECE3;
    }
    
    .header-box {
        background: #4A70A9;
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }
    
    .header-box h1 {
        margin: 0;
        font-size: 1.75rem;
        font-weight: 600;
    }
    
    .header-box p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 0.9rem;
    }
    
    .card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
    }
    
    .card-title {
        font-size: 0.85rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.5rem;
        font-weight: 500;
    }
    
    .card-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a1a1a;
    }
    
    .card-value.temp { color: #e74c3c; }
    .card-value.hum { color: #3498db; }
    .card-value.gas { color: #f39c12; }
    .card-value.safe { color: #27ae60; }
    
    .card-unit {
        font-size: 1rem;
        color: #888;
        font-weight: 400;
    }
    
    .status-badge {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    
    .status-online {
        background: #d4edda;
        color: #155724;
    }
    
    .status-offline {
        background: #f8d7da;
        color: #721c24;
    }
    
    .section-title {
        font-size: 1rem;
        font-weight: 600;
        color: #333;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #4A70A9;
    }
    
    .led-btn {
        width: 100%;
        padding: 0.75rem;
        border: none;
        border-radius: 8px;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.2s;
        margin-bottom: 0.5rem;
    }
    
    .info-row {
        display: flex;
        justify-content: space-between;
        padding: 0.5rem 0;
        border-bottom: 1px solid #eee;
        font-size: 0.9rem;
    }
    
    .info-row:last-child { border-bottom: none; }
    .info-label { color: #666; }
    .info-value { color: #333; font-weight: 500; }
    
    #MainMenu, footer, header { visibility: hidden; }
    
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        font-weight: 500;
        border: none;
        transition: all 0.2s;
        background: #4A70A9 !important;
        color: white !important;
    }
    
    .stButton > button:hover {
        background: #3d5d8a !important;
        color: white !important;
        border: none !important;
    }
    
    .stButton > button:active,
    .stButton > button:focus {
        background: #2c4a6e !important;
        color: white !important;
        border: none !important;
        box-shadow: none !important;
    }
    
    /* Primary button */
    .stButton > button[kind="primary"] {
        background: #4A70A9 !important;
        color: white !important;
    }
    
    .stButton > button[kind="primary"]:hover {
        background: #3d5d8a !important;
    }
    
    /* Download button */
    .stDownloadButton > button {
        background: #4A70A9 !important;
        color: white !important;
        border: none !important;
    }
    
    .stDownloadButton > button:hover {
        background: #3d5d8a !important;
        color: white !important;
    }
    
    /* Tabs styling - fix visibility */
    .stTabs [data-baseweb="tab-list"] {
        background-color: white;
        border-radius: 10px;
        padding: 0.25rem;
        gap: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #4A70A9 !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
        padding: 0.5rem 1rem !important;
        border-radius: 8px !important;
        background: transparent !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(74, 112, 169, 0.1) !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: #4A70A9 !important;
        color: white !important;
    }
    
    .stTabs [data-baseweb="tab-highlight"] {
        display: none !important;
    }
    
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 1rem;
    }
    
    .block-container { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

# Auto refresh
st_autorefresh(interval=2000, key="refresh")

# ==================== HEADER ====================
st.markdown("""
<div class="header-box">
    <h1>📊 ESP32 Monitoring Dashboard</h1>
    <p>Sistem monitoring suhu, kelembaban & gas dengan prediksi ML (Random Forest)</p>
</div>
""", unsafe_allow_html=True)

# ==================== LAYOUT ====================
col_left, col_right = st.columns([1, 2], gap="large")

with col_left:
    # Connection status
    is_connected = len(st.session_state.logs) > 0
    status_class = "status-online" if is_connected else "status-offline"
    status_text = "● Online" if is_connected else "○ Offline"
    
    st.markdown(f"""
    <div class="card">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="color: #666; font-size: 0.9rem;">Status Koneksi</span>
            <span class="status-badge {status_class}">{status_text}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sensor Cards
    if st.session_state.last:
        data = st.session_state.last
        
        # Suhu
        st.markdown(f"""
        <div class="card">
            <div class="card-title">🌡️ Suhu</div>
            <div class="card-value temp">{data['suhu']:.1f}<span class="card-unit">°C</span></div>
        </div>
        """, unsafe_allow_html=True)
        
        # Kelembaban
        st.markdown(f"""
        <div class="card">
            <div class="card-title">💧 Kelembaban</div>
            <div class="card-value hum">{data['lembab']:.1f}<span class="card-unit">%</span></div>
        </div>
        """, unsafe_allow_html=True)
        
        # Gas
        gas_class = "safe" if data['status'] == "Aman" else "gas"
        status_color = "#27ae60" if data['status'] == "Aman" else ("#f39c12" if data['status'] == "Waspada" else "#e74c3c")
        conf_text = f" ({data['confidence']:.1f}%)" if data.get('confidence') else ""
        
        st.markdown(f"""
        <div class="card">
            <div class="card-title">💨 Gas</div>
            <div class="card-value {gas_class}">{data['gas']}<span class="card-unit"> ppm</span></div>
        </div>
        """, unsafe_allow_html=True)
        
        # Status Prediksi ML
        st.markdown(f"""
        <div class="card">
            <div class="card-title">🤖 Prediksi ML</div>
            <div class="card-value" style="color: {status_color};">{data['status']}</div>
            <div style="margin-top: 0.5rem; font-size: 0.85rem; color: #666;">Confidence: {conf_text if conf_text else 'N/A'}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Last update
        st.markdown(f"""
        <div class="card">
            <div class="info-row">
                <span class="info-label">Update terakhir</span>
                <span class="info-value">{data['ts']}</span>
            </div>
            <div class="info-row">
                <span class="info-label">Total data</span>
                <span class="info-value">{len(st.session_state.logs)}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="card" style="text-align: center; padding: 2rem;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">📡</div>
            <div style="color: #666;">Menunggu data dari ESP32...</div>
        </div>
        """, unsafe_allow_html=True)
    
    # LED Control
    st.markdown('<div class="section-title">💡 Kontrol LED</div>', unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🟢 Hijau", use_container_width=True):
            send_led("green_on")
            st.toast("LED Hijau ON!", icon="🟢")
    with c2:
        if st.button("🟡 Kuning", use_container_width=True):
            send_led("yellow_on")
            st.toast("LED Kuning ON!", icon="🟡")
    
    c3, c4 = st.columns(2)
    with c3:
        if st.button("🔴 Merah", use_container_width=True):
            send_led("red_on")
            st.toast("LED Merah ON!", icon="🔴")
    with c4:
        if st.button("🌈 Semua", use_container_width=True, type="primary"):
            send_led("green_on")
            send_led("yellow_on")
            send_led("red_on")
            st.toast("Semua LED ON!", icon="✨")
    

with col_right:
    st.markdown('<div class="section-title">📈 Grafik Real-time</div>', unsafe_allow_html=True)
    
    if st.session_state.logs:
        df = pd.DataFrame(st.session_state.logs[-MAX_POINTS:])
        
        tab1, tab2, tab3 = st.tabs(["Suhu", "Kelembaban", "Gas"])
        
        with tab1:
            st.line_chart(df.set_index('ts')['suhu'], color="#e74c3c", height=280)
        with tab2:
            st.line_chart(df.set_index('ts')['lembab'], color="#3498db", height=280)
        with tab3:
            st.line_chart(df.set_index('ts')['gas'], color="#f39c12", height=280)
    else:
        st.info("Menunggu data untuk menampilkan grafik...")
    
    st.markdown('<div class="section-title">📋 Data Log</div>', unsafe_allow_html=True)
    
    if st.session_state.logs:
        df_display = pd.DataFrame(st.session_state.logs)[::-1].head(15)
        # Pilih kolom yang ditampilkan
        df_show = df_display[['ts', 'suhu', 'lembab', 'gas', 'status']].copy()
        df_show.columns = ['Waktu', 'Suhu (°C)', 'Kelembaban (%)', 'Gas (ppm)', 'Status ML']
        st.dataframe(df_show, use_container_width=True, height=280, hide_index=True)
        
        # Download
        csv = pd.DataFrame(st.session_state.logs).to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download CSV", csv, f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", use_container_width=True)
    else:
        st.info("Belum ada data log")

# Footer
if model:
    model_status = "✅ Model loaded"
else:
    model_status = "❌ Model error - Retrain model dengan Python 3.11"

st.markdown(f"""
<div style="text-align: center; padding: 1rem; color: #888; font-size: 0.8rem; margin-top: 2rem;">
    ESP32 IoT Dashboard • Projek Asma • {model_status}<br>
    <span style="font-size: 0.75rem; opacity: 0.7;">MQTT: {TOPIC_DATA} | {TOPIC_LED} | {TOPIC_PREDICTION}</span>
</div>
""", unsafe_allow_html=True)

process_incoming()
