import sqlite3
from datetime import datetime, timedelta, timezone

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st
import tensorflow as tf


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
)


# ==================================================
# THEME
# ==================================================

INK = "#0F1E3D"       # sidebar, headings
SLATE = "#5B6B86"     # secondary text
CANVAS = "#F4F6FA"    # page background
OK = "#0E9F8E"        # normal / safe
BAD = "#E5484D"       # fraud / risk
LINE = "#E3E8F0"      # borders

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp {{
    font-family: 'Manrope', sans-serif;
}}
.stApp {{ background: {CANVAS}; }}
.block-container {{ padding-top: 2rem; padding-bottom: 4rem; max-width: 1200px; }}

/* hide default chrome we don't need */
footer {{ visibility: hidden; }}
.stDeployButton {{ display: none; }}

/* sidebar */
[data-testid="stSidebar"] {{ background: {INK}; }}
[data-testid="stSidebar"] * {{ color: #E6ECF8; }}
[data-testid="stSidebar"] hr {{ border-color: rgba(255,255,255,.12); }}
.brand {{ display:flex; align-items:center; gap:.7rem; margin: .5rem 0 1rem; }}
.brand-mark {{ width:38px; height:38px; border-radius:10px; background:{OK};
    display:flex; align-items:center; justify-content:center; font-size:1.2rem; }}
.brand-name {{ font-weight:800; font-size:1.1rem; line-height:1.1; }}
.brand-sub {{ font-size:.75rem; color:#9FB0CF !important; }}
.side-label {{ font-size:.78rem; color:#9FB0CF !important; margin-bottom:.15rem; }}
.side-value {{ font-weight:600; margin-bottom:.8rem; }}
.side-threshold {{ background: rgba(255,255,255,.07); border-radius:12px; padding:.8rem 1rem; }}
.side-threshold .num {{ font-size:1.6rem; font-weight:800; }}

/* hero */
.hero {{ background:{INK}; border-radius:18px; padding:1.8rem 2rem; margin-bottom:1.5rem;
    border-bottom: 4px solid {OK}; }}
.hero h1 {{ color:#fff; font-size:2rem; font-weight:800; margin:0 0 .4rem; padding:0; }}
.hero p {{ color:#B7C4DE; margin:0; max-width:46rem; font-size:1rem; line-height:1.55; }}

/* section titles */
.section-title {{ font-size:1.15rem; font-weight:700; color:{INK}; margin: 1.6rem 0 .2rem; }}
.section-sub {{ color:{SLATE}; font-size:.9rem; margin-bottom:.8rem; }}

/* generic card */
.card {{ background:#fff; border:1px solid {LINE}; border-radius:14px; padding:1.1rem 1.3rem; }}

/* metric cards */
[data-testid="stMetric"] {{ background:#fff; border:1px solid {LINE}; border-radius:14px;
    padding:1rem 1.2rem; }}
[data-testid="stMetricLabel"] p {{ color:{SLATE}; font-weight:600; }}
[data-testid="stMetricValue"] {{ color:{INK}; font-weight:800; }}

/* verdict */
.verdict {{ border-radius:16px; padding:1.3rem 1.5rem; border:1px solid; border-left-width:6px; }}
.verdict.fraud {{ background:#FEF2F2; border-color:{BAD}; }}
.verdict.ok {{ background:#ECFAF7; border-color:{OK}; }}
.verdict h3 {{ margin:0 0 .35rem; font-size:1.35rem; font-weight:800; padding:0; }}
.verdict.fraud h3 {{ color:#B42318; }}
.verdict.ok h3 {{ color:#0A7266; }}
.verdict p {{ margin:.15rem 0; color:{INK}; font-size:.95rem; }}

/* architecture */
.arch {{ display:flex; align-items:center; justify-content:center; gap:.6rem; flex-wrap:wrap;
    padding:1rem 0; }}
.node {{ background:{INK}; color:#fff; border-radius:12px; width:96px;
    display:flex; flex-direction:column; align-items:center; justify-content:center; }}
.node.core {{ background:{OK}; }}
.node b {{ font-size:1.3rem; }}
.node span {{ font-size:.72rem; opacity:.85; }}
.arrow {{ color:{SLATE}; font-size:1.3rem; }}

/* buttons */
.stButton > button {{ border-radius:10px; font-weight:600; border:1px solid {LINE}; }}
.stButton > button[kind="primary"] {{ background:{OK}; border-color:{OK}; color:#fff; }}
.stButton > button[kind="primary"]:hover {{ background:#0B8678; border-color:#0B8678; }}

/* tabs */
.stTabs [data-baseweb="tab"] {{ font-weight:600; }}
.stTabs [aria-selected="true"] {{ color:{OK}; }}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


def html(markup: str):
    st.markdown(markup, unsafe_allow_html=True)


def hero(title: str, text: str):
    html(f'<div class="hero"><h1>{title}</h1><p>{text}</p></div>')


def section(title: str, sub: str = ""):
    html(
        f'<div class="section-title">{title}</div>'
        + (f'<div class="section-sub">{sub}</div>' if sub else "")
    )


def style_figure(fig, height=320):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Manrope, sans-serif", color=INK),
    )
    return fig


# ==================================================
# DATABASE
# ==================================================

DATABASE = "fraud_history.db"

# Indian Standard Time (UTC+05:30)
IST = timezone(timedelta(hours=5, minutes=30))


def now_ist() -> str:
    return datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")


def init_database():
    conn = sqlite3.connect(DATABASE)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            time REAL,
            amount REAL,
            anomaly_score REAL,
            threshold REAL,
            status TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def save_transaction(time, amount, anomaly_score, threshold, status):
    conn = sqlite3.connect(DATABASE)
    conn.execute(
        """
        INSERT INTO transactions
        (timestamp, time, amount, anomaly_score, threshold, status)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            now_ist(),
            float(time),
            float(amount),
            float(anomaly_score),
            float(threshold),
            status,
        ),
    )
    conn.commit()
    conn.close()


def get_transaction_history():
    conn = sqlite3.connect(DATABASE)
    df = pd.read_sql_query(
        """
        SELECT id, timestamp, time, amount, anomaly_score, threshold, status
        FROM transactions
        ORDER BY id DESC
        """,
        conn,
    )
    conn.close()
    return df


def clear_transaction_history():
    conn = sqlite3.connect(DATABASE)
    conn.execute("DELETE FROM transactions")
    conn.commit()
    conn.close()


init_database()


# ==================================================
# LOAD MODEL AND FILES
# ==================================================

@st.cache_resource
def load_components():
    model = tf.keras.models.load_model("fraud_autoencoder.keras")
    scaler = joblib.load("scaler.pkl")

    try:
        threshold = joblib.load("fraud_threshold_shap.pkl")
    except Exception:
        threshold = 17.48657316949157

    background = np.load("shap_background.npy")
    samples = np.load("sample_transactions.npy")

    return model, scaler, float(threshold), background, samples


autoencoder, scaler, threshold, background, samples = load_components()

feature_names = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


# ==================================================
# MODEL HELPERS
# ==================================================

def reconstruction_error(X):
    reconstructed = autoencoder.predict(X, verbose=0)
    return np.mean(np.square(X - reconstructed), axis=1)


@st.cache_resource
def create_shap_explainer(_background):
    return shap.Explainer(reconstruction_error, _background)


explainer = create_shap_explainer(background)


# ==================================================
# VISUAL COMPONENTS
# ==================================================

def verdict_card(is_fraud: bool, score: float, thr: float):
    ratio = score / thr if thr else 0
    tone = "fraud" if is_fraud else "ok"
    title = "Potential fraud detected" if is_fraud else "Transaction looks normal"
    relation = "above" if is_fraud else "below"
    action = (
        "Send this transaction for manual review."
        if is_fraud
        else "No action needed."
    )
    html(
        f'<div class="verdict {tone}">'
        f"<h3>{title}</h3>"
        f"<p>The anomaly score <b>{score:.4f}</b> is {relation} the threshold "
        f"<b>{thr:.4f}</b> ({ratio:.2f}× the threshold).</p>"
        f"<p>{action}</p>"
        "</div>"
    )


def gauge_chart(score: float, thr: float):
    axis_max = max(thr * 2, score * 1.25)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"valueformat": ".4f", "font": {"size": 34}},
            title={"text": "Anomaly score", "font": {"size": 14}},
            gauge={
                "axis": {"range": [0, axis_max]},
                "bar": {"color": BAD if score >= thr else OK, "thickness": 0.3},
                "steps": [
                    {"range": [0, thr], "color": "#DDF3EF"},
                    {"range": [thr, axis_max], "color": "#FCE3E4"},
                ],
                "threshold": {
                    "line": {"color": INK, "width": 3},
                    "thickness": 0.85,
                    "value": thr,
                },
            },
        )
    )
    return style_figure(fig, height=270)


def shap_chart(df: pd.DataFrame):
    d = df.sort_values("SHAP Value")
    colors = [BAD if v > 0 else OK for v in d["SHAP Value"]]
    fig = go.Figure(
        go.Bar(
            x=d["SHAP Value"],
            y=d["Feature"],
            orientation="h",
            marker_color=colors,
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        )
    )
    fig.update_xaxes(title="Effect on anomaly score", zeroline=True,
                     zerolinecolor=SLATE, gridcolor=LINE)
    fig.update_yaxes(title=None)
    return style_figure(fig, height=max(280, 34 * len(d) + 80))


def architecture_diagram():
    layers = [
        ("Input", 30, False),
        ("Dense", 16, False),
        ("Bottleneck", 8, True),
        ("Dense", 16, False),
        ("Output", 30, False),
    ]
    parts = []
    for i, (name, units, core) in enumerate(layers):
        height = 60 + units * 3
        cls = "node core" if core else "node"
        parts.append(
            f'<div class="{cls}" style="height:{height}px">'
            f"<b>{units}</b><span>{name}</span></div>"
        )
        if i < len(layers) - 1:
            parts.append('<div class="arrow">→</div>')
    html('<div class="arch">' + "".join(parts) + "</div>")


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:
    html(
        '<div class="brand"><div class="brand-mark">💳</div>'
        '<div><div class="brand-name">Fraud Detector</div>'
        '<div class="brand-sub">Autoencoder + SHAP</div></div></div>'
    )

    page = st.radio(
        "Navigate",
        ["Analyze transaction", "Dashboard", "Model information"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    html(
        '<div class="side-label">Algorithm</div><div class="side-value">Autoencoder</div>'
        '<div class="side-label">Detection method</div><div class="side-value">Reconstruction error</div>'
        '<div class="side-label">Explainability</div><div class="side-value">SHAP</div>'
        '<div class="side-label">Input features</div><div class="side-value">30</div>'
    )

    html(
        '<div class="side-threshold"><div class="side-label">Detection threshold</div>'
        f'<div class="num">{threshold:.4f}</div></div>'
    )


# ==================================================
# DASHBOARD PAGE
# ==================================================

if page == "Dashboard":
    hero(
        "Dashboard",
        "Every transaction you analyze is saved here, so you can track how often "
        "the model flags activity and how far scores sit from the threshold.",
    )

    history_df = get_transaction_history()
    total = len(history_df)
    fraud = int((history_df["status"] == "Potential Fraud").sum())
    normal = total - fraud
    fraud_rate = (fraud / total * 100) if total else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions analyzed", total)
    c2.metric("Potential fraud", fraud)
    c3.metric("Normal", normal)
    c4.metric("Fraud rate", f"{fraud_rate:.2f}%")

    if history_df.empty:
        st.info("No transactions yet. Open **Analyze transaction** to run your first check.")
        st.stop()

    section("Overview", "How the analyzed transactions split, and how their scores trend.")

    left, right = st.columns([1, 2])

    with left:
        donut = go.Figure(
            go.Pie(
                labels=["Normal", "Potential fraud"],
                values=[normal, fraud],
                hole=0.62,
                marker=dict(colors=[OK, BAD]),
                textinfo="percent",
                sort=False,
            )
        )
        donut.update_layout(showlegend=True, legend=dict(orientation="h", y=-0.1))
        st.plotly_chart(style_figure(donut, 320), use_container_width=True)

    with right:
        ordered = history_df.sort_values("id")
        trend = go.Figure()
        trend.add_trace(
            go.Scatter(
                x=ordered["id"],
                y=ordered["anomaly_score"],
                mode="lines+markers",
                line=dict(color=INK, width=1.5),
                marker=dict(
                    size=9,
                    color=[BAD if s == "Potential Fraud" else OK for s in ordered["status"]],
                ),
                name="Anomaly score",
                hovertemplate="Transaction %{x}<br>Score %{y:.4f}<extra></extra>",
            )
        )
        trend.add_hline(
            y=threshold,
            line_dash="dash",
            line_color=BAD,
            annotation_text="Threshold",
            annotation_position="top left",
        )
        trend.update_xaxes(title="Transaction ID", gridcolor=LINE, dtick=1)
        trend.update_yaxes(title="Anomaly score", gridcolor=LINE)
        st.plotly_chart(style_figure(trend, 320), use_container_width=True)

    section("Transaction history")

    status_filter = st.radio(
        "Show",
        ["All", "Potential Fraud", "Normal"],
        horizontal=True,
        label_visibility="collapsed",
    )

    view = history_df if status_filter == "All" else history_df[history_df["status"] == status_filter]

    display_history = view.rename(
        columns={
            "id": "ID",
            "timestamp": "Date & time (IST)",
            "time": "Time",
            "amount": "Amount",
            "anomaly_score": "Anomaly score",
            "threshold": "Threshold",
            "status": "Status",
        }
    )

    st.dataframe(display_history, use_container_width=True, hide_index=True)

    b1, b2, _ = st.columns([1, 1, 3])
    with b1:
        st.download_button(
            "Download CSV",
            data=display_history.to_csv(index=False).encode("utf-8"),
            file_name="transaction_history.csv",
            mime="text/csv",
        )
    with b2:
        with st.popover("Clear history"):
            st.write("This permanently deletes every saved transaction.")
            if st.button("Yes, delete all", type="primary"):
                clear_transaction_history()
                st.rerun()

    st.stop()


# ==================================================
# MODEL INFORMATION PAGE
# ==================================================

if page == "Model information":
    hero(
        "Model information",
        "How the autoencoder decides what is unusual, and how to read its explanations.",
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Input features", "30")
    c2.metric("Detection method", "Reconstruction error")
    c3.metric("Threshold", f"{threshold:.4f}")

    tab_overview, tab_arch, tab_shap = st.tabs(["How it works", "Architecture", "Explainability"])

    with tab_overview:
        st.write(
            "The autoencoder is trained to rebuild ordinary transactions. When it is given "
            "a transaction that looks unlike what it learned, the rebuilt version is a poor "
            "match. The average squared difference between the two is the **anomaly score**. "
            "A score at or above the threshold is flagged as potential fraud."
        )
        st.info("Time + V1–V28 + Amount make up the 30 inputs. V1–V28 are anonymized features from the original dataset.")
        st.warning(
            "The anomaly score measures how unusual a transaction is. It is not the probability of fraud."
        )

    with tab_arch:
        architecture_diagram()
        st.caption("Layer sizes: 30 → 16 → 8 → 16 → 30. The 8-unit bottleneck forces the model to learn a compact summary of normal behavior.")

    with tab_shap:
        st.write(
            "SHAP shows how much each feature pushed the anomaly score up or down for one "
            "transaction."
        )
        st.markdown(
            f"- <span style='color:{BAD}'>**Positive values**</span> raise the reconstruction error.\n"
            f"- <span style='color:{OK}'>**Negative values**</span> lower it.\n"
            "- SHAP values explain the score. They do not prove a transaction is fraudulent.",
            unsafe_allow_html=True,
        )

    st.stop()


# ==================================================
# ANALYZE TRANSACTION PAGE
# ==================================================

hero(
    "Credit card fraud detection",
    "Enter a transaction, or load a sample, and the autoencoder will score how unusual it is "
    "and show which features drove that score.",
)

# ---- sample controls ----

b1, b2, _ = st.columns([1.2, 1, 4])

with b1:
    load_clicked = st.button("Load sample transaction", use_container_width=True)
with b2:
    reset_clicked = st.button("Reset form", use_container_width=True)


def clear_form_state():
    for i in range(1, 29):
        st.session_state.pop(f"v{i}", None)
    st.session_state.pop("time_input", None)
    st.session_state.pop("amount_input", None)


if load_clicked:
    sample_scaled = samples[np.random.randint(len(samples))]
    sample_original = scaler.inverse_transform(sample_scaled.reshape(1, -1))[0]
    st.session_state["sample_data"] = sample_original
    st.session_state.pop("last_result", None)
    clear_form_state()
    st.toast("Sample transaction loaded", icon="✅")

if reset_clicked:
    st.session_state.pop("sample_data", None)
    st.session_state.pop("last_result", None)
    clear_form_state()
    st.toast("Form reset", icon="↩️")

sample_data = st.session_state.get("sample_data", None)


def default_for(index: int, fallback: float) -> float:
    return float(sample_data[index]) if sample_data is not None else fallback


# ---- inputs ----

section("Transaction details", "Time and amount are in the original dataset units.")

with st.container(border=True):
    t1, t2 = st.columns(2)
    with t1:
        time = st.number_input("Time", min_value=0.0, value=default_for(0, 0.0), key="time_input")
    with t2:
        amount = st.number_input("Amount", min_value=0.0, value=default_for(29, 100.0), key="amount_input")

section("Anonymized features", "V1–V28 come from the original credit-card dataset.")

tab_a, tab_b = st.tabs(["V1 – V14", "V15 – V28"])

v_values = {}

with tab_a:
    cols = st.columns(4)
    for i in range(1, 15):
        with cols[(i - 1) % 4]:
            v_values[i] = st.number_input(
                f"V{i}", value=default_for(i, 0.0), key=f"v{i}", format="%.4f"
            )

with tab_b:
    cols = st.columns(4)
    for i in range(15, 29):
        with cols[(i - 15) % 4]:
            v_values[i] = st.number_input(
                f"V{i}", value=default_for(i, 0.0), key=f"v{i}", format="%.4f"
            )

transaction = [time] + [v_values[i] for i in range(1, 29)] + [amount]

if len(transaction) != 30:
    st.error(f"Expected 30 features, but received {len(transaction)}.")
    st.stop()

transaction_array = np.array(transaction).reshape(1, -1)

st.write("")
analyze_clicked = st.button("Analyze transaction", type="primary", use_container_width=True)


# ---- run analysis ----

if analyze_clicked:
    transaction_scaled = scaler.transform(transaction_array)
    score = float(reconstruction_error(transaction_scaled)[0])
    is_fraud = score >= threshold
    prediction = "Potential Fraud" if is_fraud else "Normal"

    save_transaction(
        time=time,
        amount=amount,
        anomaly_score=score,
        threshold=threshold,
        status=prediction,
    )

    with st.spinner("Calculating SHAP explanation..."):
        shap_result = explainer(transaction_scaled)

    values = shap_result.values[0]
    explanation = (
        pd.DataFrame(
            {
                "Feature": feature_names,
                "SHAP Value": values,
                "Absolute SHAP": np.abs(values),
            }
        )
        .sort_values("Absolute SHAP", ascending=False)
        .reset_index(drop=True)
    )

    st.session_state["last_result"] = {
        "score": score,
        "threshold": threshold,
        "is_fraud": bool(is_fraud),
        "explanation": explanation,
    }


# ---- show result (persists while the user adjusts the chart) ----

result = st.session_state.get("last_result")

if result:
    section("Detection result")

    left, right = st.columns([3, 2])
    with left:
        verdict_card(result["is_fraud"], result["score"], result["threshold"])
    with right:
        st.plotly_chart(gauge_chart(result["score"], result["threshold"]), use_container_width=True)

    section("Why this score?", "SHAP shows which features pushed the anomaly score up or down.")

    top_n = st.slider("Features to show", min_value=3, max_value=15, value=8)
    top = result["explanation"].head(top_n)

    leader = top.iloc[0]
    direction = "raised" if leader["SHAP Value"] > 0 else "lowered"
    st.markdown(
        f"**{leader['Feature']}** had the largest influence: it {direction} the anomaly "
        f"score by **{abs(leader['SHAP Value']):.4f}**."
    )

    chart_col, table_col = st.columns([3, 2])
    with chart_col:
        st.plotly_chart(shap_chart(top), use_container_width=True)
    with table_col:
        table = top[["Feature", "SHAP Value"]].copy()
        table["SHAP Value"] = table["SHAP Value"].round(4)
        st.dataframe(table, use_container_width=True, hide_index=True)

    st.caption(
        "Red bars raise the reconstruction error, green bars lower it. SHAP values explain the "
        "anomaly score. They are not a fraud probability and do not prove fraud."
    )