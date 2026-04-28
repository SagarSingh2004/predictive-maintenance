import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go

# --- Training Data Ranges ---
RANGES = {
    "air_temp": (295.3, 304.5),
    "process_temp": (305.7, 313.8),
    "rpm": (1168, 2886),
    "torque": (3.8, 76.6),
    "tool_wear": (0, 253)
}

# --- Page Config ---
st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif !important;
    }
    
    /* Main Background */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%);
    }
    
    /* Sidebar Background */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1423 0%, #1a1f3a 100%);
    }
    
    /* Sidebar Card Styling */
    .sidebar-card {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.08) 0%, rgba(0, 153, 255, 0.04) 100%);
        border: 1px solid rgba(0, 212, 255, 0.25);
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 15px;
        backdrop-filter: blur(10px);
    }
    
    .sidebar-card:hover {
        border-color: rgba(0, 212, 255, 0.5);
        box-shadow: 0 5px 20px rgba(0, 212, 255, 0.1);
    }
    
    .sidebar-card-title {
        color: #00d4ff;
        font-weight: 700;
        font-size: 1.1em;
        margin-bottom: 12px;
    }
    
    .sidebar-card-content {
        color: #e0e0ff;
        font-size: 0.95em;
        line-height: 1.8;
    }
    
    /* Hide default form border */
    [data-testid="stForm"] {
        border: none !important;
    }
    
    /* Input styling */
    input {
        background: rgba(255, 255, 255, 0.95) !important;
        color: #333 !important;
        border: 2px solid #00d4ff !important;
        border-radius: 10px !important;
        font-weight: 500 !important;
        font-family: 'Poppins' !important;
    }
    
    input:focus {
        border: 2px solid #0099ff !important;
        box-shadow: 0 0 15px rgba(0, 212, 255, 0.4) !important;
    }
    
    /* Select box styling */
    [role="listbox"] {
        background: rgba(255, 255, 255, 0.95) !important;
        border: 2px solid #00d4ff !important;
        border-radius: 10px !important;
    }
    
    select {
        background: rgba(255, 255, 255, 0.95) !important;
        color: #333 !important;
        border: 2px solid #00d4ff !important;
        border-radius: 10px !important;
        font-weight: 500 !important;
    }
    
    /* Text color adjustments */
    label {
        color: #00d4ff !important;
        font-weight: 600 !important;
    }
    
    /* Button styling */
    button {
        background: linear-gradient(90deg, #00d4ff, #0099ff) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 24px !important;
        font-weight: 700 !important;
        transition: all 0.3s !important;
    }
    
    button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 30px rgba(0, 212, 255, 0.4) !important;
    }
    
    /* Card styling */
    .metric-card {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, rgba(0, 153, 255, 0.05) 100%);
        border: 2px solid rgba(0, 212, 255, 0.3);
        border-radius: 15px;
        padding: 25px;
        text-align: center;
        transition: all 0.3s;
    }
    
    .metric-card:hover {
        border-color: rgba(0, 212, 255, 0.6);
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(0, 212, 255, 0.2);
    }
    
    .metric-label {
        color: #a0a0ff;
        font-size: 0.9em;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 10px;
        font-weight: 600;
    }
    
    .metric-value {
        font-size: 2.5em;
        font-weight: 800;
        color: #00d4ff;
    }
    
    /* Alert styling */
    .alert-box {
        padding: 20px;
        border-radius: 12px;
        margin: 20px 0;
        border-left: 5px solid;
        font-weight: 500;
    }
    
    .alert-safe {
        background: rgba(0, 208, 132, 0.1);
        color: #00d084;
        border-left-color: #00d084;
    }
    
    .alert-monitor {
        background: rgba(255, 193, 7, 0.1);
        color: #ffc107;
        border-left-color: #ffc107;
    }
    
    .alert-warning {
        background: rgba(255, 149, 0, 0.1);
        color: #ff9500;
        border-left-color: #ff9500;
    }
    
    .alert-critical {
        background: rgba(255, 59, 48, 0.1);
        color: #ff3b30;
        border-left-color: #ff3b30;
    }
    
    /* Text colors */
    h1, h2, h3 { color: #ffffff !important; }
    p { color: #e0e0ff !important; }
    
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    # Card 1: Project Info
    st.markdown("""
    <div class="sidebar-card">
        <div class="sidebar-card-title">🔧 PREDICTIVE MAINTENANCE</div>
        <div class="sidebar-card-content">
            <b>Capstone Project</b><br>
            Author: Sagar Singh<br>
            Date: 2026-04-24
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Card 2: About
    st.markdown("""
    <div class="sidebar-card">
        <div class="sidebar-card-title">📊 About</div>
        <div class="sidebar-card-content">
            ML-powered system to predict machine failure risk in real-time using advanced algorithms.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Card 3: Risk Levels
    st.markdown("""
    <div class="sidebar-card">
        <div class="sidebar-card-title">🎯 Risk Levels</div>
        <div class="sidebar-card-content">
            🟢 <b>Safe:</b> &lt; 20%<br>
            🟡 <b>Monitor:</b> 20-49%<br>
            🟠 <b>Warning:</b> 50-69%<br>
            🔴 <b>Critical:</b> ≥ 70%
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Card 4: Valid Ranges
    st.markdown("""
    <div class="sidebar-card">
        <div class="sidebar-card-title">✅ Valid Ranges</div>
        <div class="sidebar-card-content">
            • <b>RPM:</b> 1,168 - 2,886<br>
            • <b>Air Temp:</b> 295.3 - 304.5 K<br>
            • <b>Process Temp:</b> 305.7 - 313.8 K<br>
            • <b>Torque:</b> 3.8 - 76.6 Nm<br>
            • <b>Tool Wear:</b> 0 - 253 min
        </div>
    </div>
    """, unsafe_allow_html=True)

# Load Model
try:
    loaded = joblib.load("voting_model.pkl")
    model = loaded["model"]
    threshold = loaded["threshold"]
except:
    st.error("❌ Model file not found!")
    st.stop()

def get_risk_info(prob):
    if prob < 0.2:
        return "Safe", "🟢", "#00d084", "✅ SAFE CONDITION"
    elif prob < 0.499:
        return "Monitor", "🟡", "#ffc107", "🔍 MONITOR CONDITION"
    elif prob < 0.7:
        return "Warning", "🟠", "#ff9500", "⚠️ WARNING"
    else:
        return "Critical", "🔴", "#ff3b30", "🚨 CRITICAL ALERT"

# --- MAIN CONTENT ---
col_title = st.columns([1])[0]
with col_title:
    st.markdown("""
    <h1 style='text-align: center; color: #00d4ff; font-size: 3.5em; margin-bottom: 5px;'>
    🔧 PREDICTIVE MAINTENANCE SYSTEM
    </h1>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <p style='text-align: center; color: #a0a0ff; font-size: 1.2em; margin-bottom: 20px;'>
    Advanced Machine Learning Powered Failure Prediction
    </p>
    """, unsafe_allow_html=True)

st.divider()

# --- INPUT SECTION ---
st.markdown("### ⚙️ MACHINE PARAMETERS")

with st.form("prediction_form"):
    # Row 1
    col1, col2, col3 = st.columns(3, gap="medium")
    
    with col1:
        st.markdown("**🤖 Machine Type**")
        type_val = st.selectbox("Type", ["L", "M", "H"], label_visibility="collapsed")
    
    with col2:
        st.markdown("**🌡️ Air Temperature (K)**")
        air_temp = st.number_input("Air Temp", value=300.0, step=0.1, label_visibility="collapsed")
    
    with col3:
        st.markdown("**🔥 Process Temperature (K)**")
        process_temp = st.number_input("Process Temp", value=310.0, step=0.1, label_visibility="collapsed")
    
    # Row 2
    col4, col5, col6 = st.columns(3, gap="medium")
    
    with col4:
        st.markdown("**⚡ Rotational Speed (RPM)**")
        rpm = st.number_input("RPM", value=1500, label_visibility="collapsed")
    
    with col5:
        st.markdown("**💪 Torque (Nm)**")
        torque = st.number_input("Torque", value=40.0, step=0.1, label_visibility="collapsed")
    
    with col6:
        st.markdown("**🛠️ Tool Wear (min)**")
        tool_wear = st.number_input("Tool Wear", value=100, label_visibility="collapsed")
    
    st.markdown("")
    submitted = st.form_submit_button("🚀 PREDICT FAILURE RISK", use_container_width=True)

# --- RESULTS ---
if submitted:
    # Feature Engineering
    temp_diff = process_temp - air_temp
    power_kw = (torque * (2 * np.pi * rpm / 60)) / 1000
    
    # Create DataFrame
    input_df = pd.DataFrame([{
        "type": type_val,
        "air_temperature_k": air_temp,
        "process_temperature_k": process_temp,
        "rotational_speed_rpm": rpm,
        "torque_nm": torque,
        "tool_wear_min": tool_wear,
        "temp_diff": temp_diff,
        "power_kw": power_kw
    }])
    
    # --- Out-of-Distribution Check ---
    warnings = []
    if not (RANGES["air_temp"][0] <= air_temp <= RANGES["air_temp"][1]):
        warnings.append("Air Temperature outside training range")
    if not (RANGES["process_temp"][0] <= process_temp <= RANGES["process_temp"][1]):
        warnings.append("Process Temperature outside training range")
    if not (RANGES["rpm"][0] <= rpm <= RANGES["rpm"][1]):
        warnings.append("RPM outside training range")
    if not (RANGES["torque"][0] <= torque <= RANGES["torque"][1]):
        warnings.append("Torque outside training range")
    if not (RANGES["tool_wear"][0] <= tool_wear <= RANGES["tool_wear"][1]):
        warnings.append("Tool Wear outside training range")
    
    # Prediction
    probs = model.predict_proba(input_df)[:, 1][0]
    pred = int(probs > threshold)
    
    # --- Domain Rule Override (Improved) ---
    safety_critical = False
    domain_warning = ""
    
    if process_temp > 320:
        safety_critical = True
        domain_warning = "Process Temperature exceeds safe operating limit"
    
    if torque > 70:
        safety_critical = True
        if domain_warning:
            domain_warning += " AND Torque exceeds safe operating limit"
        else:
            domain_warning = "Torque exceeds safe operating limit"
    
    if safety_critical:
        probs = max(probs, 0.85)
        pred = 1
    
    risk_label, risk_emoji, risk_color, risk_msg = get_risk_info(probs)
    
    st.divider()
    
    # --- Show Safety Critical Warning ---
    if safety_critical:
        st.error(f"🚨 CRITICAL SAFETY ALERT: {domain_warning}")
        st.write("⚠️ Failure probability forced to 85% due to safety constraints")
        st.write("")
    
    # --- Show Out-of-Distribution Warning ---
    if warnings:
        st.warning("⚠️ One or more inputs outside training distribution:")
        for w in warnings:
            st.write(f"  • {w}")
        st.write("💡 Prediction may be less reliable — use with caution")
        st.write("")
    
    st.markdown("### 📊 PREDICTION RESULTS")
    
    # Metrics Row
    m1, m2, m3 = st.columns(3, gap="large")
    
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Failure Probability</div>
            <div class="metric-value" style="color: {risk_color};">{probs*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with m2:
        status = "🚨 FAILURE" if pred == 1 else "✅ NO FAILURE"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Prediction</div>
            <div class="metric-value">{status}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Risk Level</div>
            <div class="metric-value">{risk_emoji}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Alert Message
    alert_class = "alert-safe" if probs < 0.2 else ("alert-monitor" if probs < 0.499 else ("alert-warning" if probs < 0.7 else "alert-critical"))
    
    if probs < 0.2:
        msg = "Your machine is operating normally. No immediate action required."
    elif probs < 0.499:
        msg = "Machine is stable but monitor closely. Plan maintenance in 2-3 weeks."
    elif probs < 0.7:
        msg = "Schedule maintenance soon (within 1 week) to prevent failure."
    else:
        msg = "URGENT: Take machine offline immediately for preventive maintenance!"
    
    st.markdown(f"""
    <div class="alert-box {alert_class}">
        <strong>{risk_msg}</strong><br>
        {msg}
    </div>
    """, unsafe_allow_html=True)
    
    # --- Confidence Insight ---
    if probs > 0.8 or probs < 0.2:
        st.success("✅ High confidence prediction - model certainty is strong")
    else:
        st.info("ℹ️ Moderate confidence — monitor closely and recheck regularly")
    
    # Progress Visualization
    st.markdown("### 📈 Risk Progress")
    
    fig = go.Figure(data=[
        go.Bar(
            x=[probs],
            orientation='h',
            marker=dict(
                color=[risk_color],
                line=dict(color='#00d4ff', width=2)
            ),
            text=[f'{probs*100:.1f}%'],
            textposition='outside',
            hovertemplate='<b>Risk: %{x:.1%}</b><extra></extra>'
        )
    ])
    
    fig.update_layout(
        xaxis=dict(range=[0, 1], tickformat='.0%'),
        yaxis=dict(visible=False),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=120,
        margin=dict(l=0, r=80, t=0, b=0),
        font=dict(color='#a0a0ff')
    )
    
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    # Input Summary
    with st.expander("📋 View Input Details"):
        summary = pd.DataFrame({
            "Parameter": ["Machine Type", "Air Temp (K)", "Process Temp (K)", "RPM", "Torque (Nm)", "Tool Wear (min)", "Temp Diff (K)", "Power (kW)"],
            "Value": [type_val, f"{air_temp:.2f}", f"{process_temp:.2f}", f"{rpm}", f"{torque:.2f}", f"{tool_wear}", f"{temp_diff:.2f}", f"{power_kw:.4f}"]
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)

st.divider()
st.markdown("<p style='text-align: center; color: #a0a0ff; font-size: 0.9em;'>❤️ Made with Streamlit | Capstone Project 2026</p>", unsafe_allow_html=True)