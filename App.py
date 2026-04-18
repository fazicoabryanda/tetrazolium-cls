import streamlit as st
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO
import os
import torch
import pandas as pd
import plotly.graph_objects as go
from werkzeug.utils import secure_filename


# ═══════════════════════════════════════════════════════════════
#  1. PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="TZ Viability Test — Seed Analysis",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ═══════════════════════════════════════════════════════════════
#  2. DESIGN SYSTEM — Professional Theme CSS
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ─── Web Fonts ─── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

/* ─── Design Tokens ─── */
:root {
    --bg:             #f8faf9;
    --surface:        #ffffff;
    --surface-alt:    #f0f5f1;
    --surface-hover:  #e8f0ea;
    --border:         #d5e2d8;
    --border-light:   #e8efe9;

    --emerald-50:     #ecfdf5;
    --emerald-100:    #d1fae5;
    --emerald-200:    #a7f3d0;
    --emerald-500:    #10b981;
    --emerald-600:    #059669;
    --emerald-700:    #047857;
    --emerald-800:    #065f46;
    --emerald-900:    #064e3b;
    --emerald-950:    #022c22;

    --red-50:         #fef2f2;
    --red-100:        #fee2e2;
    --red-500:        #ef4444;
    --red-600:        #dc2626;
    --red-700:        #b91c1c;

    --amber-50:       #fffbeb;
    --amber-500:      #f59e0b;
    --amber-600:      #d97706;

    --text-primary:   #0f1f15;
    --text-secondary: #3d5a45;
    --text-muted:     #6b8a72;
    --text-faint:     #94b39c;

    --radius-sm:      8px;
    --radius-md:      12px;
    --radius-lg:      16px;
    --radius-xl:      20px;

    --shadow-sm:      0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
    --shadow-md:      0 4px 12px rgba(0,0,0,0.06), 0 2px 4px rgba(0,0,0,0.04);
    --shadow-lg:      0 10px 32px rgba(0,0,0,0.08), 0 4px 8px rgba(0,0,0,0.04);
    --shadow-accent:  0 8px 28px rgba(5,150,105,0.2);
}

/* ─── Base Reset ─── */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--text-primary) !important;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

#MainMenu, footer, header { visibility: hidden; }

.block-container {
    padding: 2rem 3rem 5rem 3rem !important;
    max-width: 1360px;
}

/* ─── Sidebar ─── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--emerald-950) 0%, var(--emerald-900) 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
[data-testid="stSidebar"] * {
    color: rgba(209,250,229,0.85) !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] strong,
[data-testid="stSidebar"] b {
    color: #ffffff !important;
}
[data-testid="stSidebar"] .stTextInput input {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    color: #ffffff !important;
    border-radius: var(--radius-sm) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
}
[data-testid="stSidebar"] .stTextInput input:focus {
    border-color: var(--emerald-500) !important;
    box-shadow: 0 0 0 2px rgba(16,185,129,0.25) !important;
}
[data-testid="stSidebar"] .stSlider > div > div > div {
    background: var(--emerald-500) !important;
}

/* ─── Hero Banner ─── */
.hero-banner {
    background: linear-gradient(135deg, var(--emerald-950) 0%, var(--emerald-800) 40%, var(--emerald-700) 100%);
    border-radius: var(--radius-xl);
    padding: 2.8rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    box-shadow: var(--shadow-lg), var(--shadow-accent);
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 260px; height: 260px;
    border-radius: 50%;
    background: rgba(16,185,129,0.12);
    pointer-events: none;
}
.hero-banner::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 30%;
    width: 180px; height: 180px;
    border-radius: 50%;
    background: rgba(167,243,208,0.06);
    pointer-events: none;
}
.hero-overline {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 0.25em;
    color: var(--emerald-200);
    text-transform: uppercase;
    margin-bottom: 0.75rem;
    opacity: 0.9;
}
.hero-heading {
    font-family: 'Inter', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    color: #ffffff;
    line-height: 1.15;
    margin: 0 0 0.75rem 0;
    letter-spacing: -0.02em;
}
.hero-heading em {
    font-style: normal;
    font-weight: 300;
    color: var(--emerald-200);
}
.hero-description {
    font-size: 0.9rem;
    font-weight: 400;
    color: rgba(255,255,255,0.65);
    max-width: 580px;
    line-height: 1.75;
}

/* ─── Status Banners ─── */
.status-success {
    background: var(--emerald-50);
    border: 1px solid var(--emerald-200);
    border-radius: var(--radius-md);
    padding: 0.85rem 1.25rem;
    display: flex;
    align-items: center;
    gap: 0.65rem;
    font-size: 0.85rem;
    margin-bottom: 1.5rem;
    color: var(--emerald-800);
}
.status-error {
    background: var(--red-50);
    border: 1px solid var(--red-100);
    border-radius: var(--radius-md);
    padding: 0.85rem 1.25rem;
    font-size: 0.85rem;
    margin-bottom: 1.5rem;
    color: var(--red-700);
}
.dot-live {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--emerald-500);
    flex-shrink: 0;
    box-shadow: 0 0 6px rgba(16,185,129,0.5);
    animation: pulse-dot 2s infinite;
}
@keyframes pulse-dot {
    0%, 100% { box-shadow: 0 0 4px rgba(16,185,129,0.4); }
    50%      { box-shadow: 0 0 10px rgba(16,185,129,0.7); }
}

/* ─── Section Headers ─── */
.section-overline {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    color: var(--text-muted);
    text-transform: uppercase;
    margin-bottom: 0.15rem;
}
.section-title {
    font-family: 'Inter', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 1rem;
    letter-spacing: -0.01em;
}

/* ─── Result Badge ─── */
.result-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.6rem 1.3rem;
    border-radius: var(--radius-sm);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.88rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    margin: 0.4rem 0 0.75rem 0;
    transition: transform 0.15s ease;
}
.result-badge:hover { transform: translateY(-1px); }

.badge-viable {
    background: var(--emerald-50);
    border: 1.5px solid var(--emerald-500);
    color: var(--emerald-700);
}
.badge-nonviable {
    background: var(--red-50);
    border: 1.5px solid var(--red-500);
    color: var(--red-700);
}
.badge-unknown {
    background: var(--amber-50);
    border: 1.5px solid var(--amber-500);
    color: var(--amber-600);
}

/* ─── Confidence Bar ─── */
.conf-wrap { margin: 0.75rem 0 1.1rem 0; }
.conf-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 0.45rem;
}
.conf-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    font-weight: 500;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.12em;
}
.conf-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    font-weight: 700;
}
.conf-track {
    background: var(--surface-alt);
    border-radius: 99px;
    height: 10px;
    width: 100%;
    overflow: hidden;
    border: 1px solid var(--border-light);
}
.conf-fill-viable {
    height: 100%; border-radius: 99px;
    background: linear-gradient(90deg, var(--emerald-600), var(--emerald-500));
}
.conf-fill-nonviable {
    height: 100%; border-radius: 99px;
    background: linear-gradient(90deg, var(--red-700), var(--red-500));
}
.conf-fill-unknown {
    height: 100%; border-radius: 99px;
    background: linear-gradient(90deg, var(--amber-600), var(--amber-500));
}

/* ─── Stat Cards ─── */
.stat-row {
    display: flex;
    gap: 0.65rem;
    flex-wrap: wrap;
    margin: 0.75rem 0;
}
.stat-card {
    background: var(--surface);
    border: 1px solid var(--border-light);
    border-radius: var(--radius-md);
    padding: 0.85rem 1rem;
    text-align: center;
    min-width: 90px;
    flex: 1;
    box-shadow: var(--shadow-sm);
    transition: box-shadow 0.2s ease, transform 0.15s ease;
}
.stat-card:hover {
    box-shadow: var(--shadow-md);
    transform: translateY(-1px);
}
.stat-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1.2;
}
.stat-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.62rem;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.25rem;
}

/* ─── Info Box ─── */
.info-box {
    background: var(--emerald-50);
    border-left: 3px solid var(--emerald-600);
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    padding: 0.9rem 1.15rem;
    margin: 0.75rem 0;
    font-size: 0.84rem;
    color: var(--emerald-800);
    line-height: 1.7;
}

/* ─── Divider ─── */
.tz-divider {
    border: none;
    border-top: 1px solid var(--border-light);
    margin: 2rem 0;
}

/* ─── Image Meta ─── */
.img-meta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    font-weight: 500;
    color: var(--text-faint);
    margin-top: 0.35rem;
}

/* ─── File Uploader ─── */
.stFileUploader > div {
    background: var(--surface) !important;
    border: 2px dashed var(--border) !important;
    border-radius: var(--radius-md) !important;
    transition: border-color 0.2s ease !important;
}
.stFileUploader > div:hover {
    border-color: var(--emerald-500) !important;
}

/* ─── Buttons ─── */
.stButton > button {
    background: linear-gradient(135deg, var(--emerald-600), var(--emerald-700)) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 0.65rem 2rem !important;
    box-shadow: var(--shadow-accent) !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.01em !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, var(--emerald-700), var(--emerald-800)) !important;
    box-shadow: 0 10px 32px rgba(5,150,105,0.3) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* ─── DataFrame ─── */
div[data-testid="stDataFrame"] {
    border: 1px solid var(--border-light) !important;
    border-radius: var(--radius-md) !important;
    overflow: hidden;
}

/* ─── Scrollbar ─── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--surface-alt); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-faint); }

/* ─── Footer ─── */
.app-footer {
    text-align: center;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    font-weight: 500;
    color: var(--text-faint);
    padding: 1rem 0 1.5rem 0;
    letter-spacing: 0.04em;
}
.app-footer a {
    color: var(--emerald-600);
    text-decoration: none;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  3. CONSTANTS & MODEL
# ═══════════════════════════════════════════════════════════════
MODEL_PATH = "tetrazolium_model.pt"
UPLOAD_DIR = "tz_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@st.cache_resource
def load_model(path: str):
    """Load YOLO classification model with safe torch patching."""
    if not os.path.exists(path):
        return None, f"Model file not found: `{path}`"
    original_load = torch.load
    def patched_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return original_load(*args, **kwargs)
    torch.load = patched_load
    try:
        model = YOLO(path)
        return model, None
    except Exception as e:
        return None, str(e)
    finally:
        torch.load = original_load


# ═══════════════════════════════════════════════════════════════
#  4. HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════
def classify_label(name: str) -> str:
    """Map raw class name to a canonical label type."""
    n = name.lower()
    if "nonviable" in n or "non_viable" in n or "non-viable" in n:
        return "nonviable"
    if "viable" in n:
        return "viable"
    return "unknown"


def confidence_bar_html(pct: float, label_type: str) -> str:
    """Render an HTML confidence bar for a given percentage and label type."""
    style_map = {
        "viable":    ("conf-fill-viable",    "#059669"),
        "nonviable": ("conf-fill-nonviable",  "#dc2626"),
    }
    fill_class, color = style_map.get(label_type, ("conf-fill-unknown", "#d97706"))
    safe_pct = min(max(float(pct), 0.0), 100.0)

    return (
        '<div class="conf-wrap">'
        '  <div class="conf-header">'
        '    <span class="conf-label">Confidence Score</span>'
        f'   <span class="conf-value" style="color:{color};">{safe_pct:.2f}%</span>'
        '  </div>'
        '  <div class="conf-track">'
        f'    <div class="{fill_class}" style="width:{safe_pct:.2f}%;"></div>'
        '  </div>'
        '</div>'
    )


def build_gauge(value: float, color: str) -> go.Figure:
    """Build a Plotly gauge chart for confidence display."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(value * 100, 2),
        number={
            "suffix": "%",
            "font": {"size": 28, "color": "#0f1f15", "family": "JetBrains Mono"},
        },
        gauge={
            "axis": {
                "range": [0, 100],
                "tickcolor": "#94b39c",
                "tickfont": {"color": "#6b8a72", "size": 9},
            },
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "#f0f5f1",
            "bordercolor": "#d5e2d8",
            "borderwidth": 1,
            "steps": [
                {"range": [0, 50],  "color": "#f8faf9"},
                {"range": [50, 75], "color": "#ecfdf5" if color == "#059669" else "#fef2f2"},
                {"range": [75, 100], "color": "#d1fae5" if color == "#059669" else "#fee2e2"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.85,
                "value": value * 100,
            },
        },
        title={
            "text": "Confidence Level",
            "font": {"color": "#6b8a72", "size": 10, "family": "Inter"},
        },
    ))
    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=40, b=5),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#0f1f15",
    )
    return fig


def build_prob_bar(probs: dict, class_names: dict) -> go.Figure:
    """Build a horizontal probability bar chart for all classes."""
    labels, values, colors = [], [], []
    color_map = {
        "viable":    "#059669",
        "nonviable": "#dc2626",
        "unknown":   "#d97706",
    }
    for idx, prob in sorted(probs.items(), key=lambda x: -x[1]):
        name = class_names.get(idx, f"Class {idx}")
        labels.append(name)
        values.append(round(prob * 100, 3))
        colors.append(color_map.get(classify_label(name), "#6b8a72"))

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=colors,
        marker_line_color="rgba(0,0,0,0.04)",
        marker_line_width=1,
        text=[f"{v:.2f}%" for v in values],
        textposition="outside",
        textfont={"color": "#0f1f15", "size": 12, "family": "JetBrains Mono"},
        hovertemplate="%{y}: %{x:.3f}%<extra></extra>",
    ))
    fig.update_layout(
        height=max(150, len(labels) * 55),
        margin=dict(l=10, r=80, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            range=[0, 120],
            showgrid=True,
            gridcolor="#e8efe9",
            tickfont={"color": "#6b8a72"},
            zeroline=False,
        ),
        yaxis=dict(tickfont={"color": "#0f1f15", "size": 13, "family": "Inter"}),
        bargap=0.35,
    )
    return fig


# ═══════════════════════════════════════════════════════════════
#  5. SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    # ── Sidebar Header
    st.markdown("""
    <div style="padding:1.3rem 0 0.5rem 0;">
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.58rem;
                    font-weight:600;letter-spacing:0.25em;color:rgba(167,243,208,0.7);
                    text-transform:uppercase;margin-bottom:0.5rem;">
            Seed Science Lab
        </div>
        <div style="font-family:'Inter',sans-serif;font-size:1.15rem;
                    font-weight:700;color:#ffffff;letter-spacing:-0.01em;">
            TZ Test Config
        </div>
    </div>
    <hr style="border-color:rgba(255,255,255,0.08);margin:0.6rem 0 1.3rem 0;">
    """, unsafe_allow_html=True)

    # ── Model Path
    st.markdown("**Model Path**")
    model_path_input = st.text_input(
        "Model path (.pt)", value=MODEL_PATH,
        help="Path ke file model klasifikasi Tetrazolium YOLOv11",
        label_visibility="collapsed",
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Confidence Threshold
    st.markdown("**Confidence Threshold**")
    conf_thresh = st.slider(
        "conf", min_value=0.01, max_value=0.99,
        value=0.25, step=0.01,
        label_visibility="collapsed",
        help="Hasil di bawah threshold ini ditandai sebagai 'low confidence'.",
    )
    st.caption(f"Threshold aktif: `{conf_thresh:.2f}`")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Batch Mode
    st.markdown("**Batch Mode**")
    batch_mode = st.toggle(
        "Aktifkan multi-gambar", value=False,
        help="Upload beberapa gambar sekaligus untuk klasifikasi batch.",
    )

    # ── Sidebar Info Card
    st.markdown("""
    <hr style="border-color:rgba(255,255,255,0.08);margin:1.5rem 0 1rem 0;">
    <div style="font-size:0.75rem;color:rgba(209,250,229,0.7);line-height:1.85;">
        <b style="color:#ffffff;font-size:0.82rem;letter-spacing:-0.01em;">
            Tetrazolium (TZ) Test
        </b><br><br>
        Uji viabilitas benih menggunakan garam tetrazolium
        (2,3,5-triphenyltetrazolium chloride). Benih <em>viable</em>
        berwarna merah karena aktivitas enzim dehidrogenase.<br><br>
        <span style="color:#10b981;">&#9679; Viable</span>
        — jaringan aktif, berwarna merah<br>
        <span style="color:#f87171;">&#9679; Non-Viable</span>
        — jaringan mati, tidak berwarna
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  6. LOAD MODEL
# ═══════════════════════════════════════════════════════════════
model, model_err = load_model(model_path_input)


# ═══════════════════════════════════════════════════════════════
#  7. HERO BANNER
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-banner">
    <div class="hero-overline">&#129516; Seed Science &amp; Technology Lab</div>
    <h1 class="hero-heading" style="color: white;">Tetrazolium Viability Test</h1>
    <p class="hero-description">
        Klasifikasi viabilitas benih secara otomatis menggunakan deep learning
        YOLOv11. Upload gambar hasil pewarnaan TZ untuk menentukan apakah benih
        <strong>viable</strong> atau <strong>non-viable</strong> beserta skor
        kepercayaannya.
    </p>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  8. MODEL STATUS
# ═══════════════════════════════════════════════════════════════
if model_err:
    st.markdown(
        '<div class="status-error">'
        '<strong>&#10007; Model Error:</strong>&nbsp;'
        f'<code>{model_err}</code><br>'
        '<span style="font-size:0.8rem;margin-top:0.3rem;display:block;opacity:0.8;">'
        'Pastikan file <code>tetrazolium_model.pt</code> berada di direktori '
        'yang sama dengan script ini, atau ubah path model di sidebar.'
        '</span></div>',
        unsafe_allow_html=True,
    )
else:
    class_names = getattr(model, "names", {})
    class_list_html = " &nbsp;&middot;&nbsp; ".join(
        f'<code style="background:var(--emerald-50);padding:2px 7px;'
        f'border-radius:5px;font-size:0.78rem;font-family:JetBrains Mono,monospace;'
        f'color:var(--emerald-800);">{v}</code>'
        for v in class_names.values()
    ) if class_names else "&mdash;"
    st.markdown(
        '<div class="status-success">'
        '<span class="dot-live"></span>'
        '<span style="font-weight:600;color:var(--emerald-800);">Model berhasil dimuat</span>'
        '<span style="color:var(--border);margin:0 0.5rem;">|</span>'
        f'<span style="font-size:0.8rem;color:var(--text-muted);">Kelas: {class_list_html}</span>'
        '</div>',
        unsafe_allow_html=True,
    )

st.markdown("<hr class='tz-divider'>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  9. FILE UPLOADER
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div class="section-overline">Input</div>
<div class="section-title">Upload Gambar Hasil Uji TZ</div>
""", unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Drag & drop gambar hasil uji Tetrazolium di sini",
    type=["png", "jpg", "jpeg", "bmp", "tiff"],
    accept_multiple_files=batch_mode,
    key="tz_uploader",
    label_visibility="collapsed",
)

if not batch_mode and uploaded is not None:
    uploaded = [uploaded]

if not uploaded:
    st.markdown("""
    <div class="info-box">
        &#128204; Upload gambar benih setelah proses pewarnaan tetrazolium.
        Format yang didukung: <strong>PNG, JPG, JPEG, BMP, TIFF</strong>.<br>
        Aktifkan <strong>Batch Mode</strong> di sidebar untuk mengklasifikasi
        banyak gambar sekaligus.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

if model is None:
    st.error("Model belum berhasil dimuat. Periksa konfigurasi di sidebar.")
    st.stop()


# ═══════════════════════════════════════════════════════════════
#  10. CLASSIFY BUTTON
# ═══════════════════════════════════════════════════════════════
run_btn = st.button(
    f"\U0001f52c  Klasifikasi {'Semua Gambar' if len(uploaded) > 1 else 'Gambar'}",
    use_container_width=True,
    type="primary",
)

if not run_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    preview_cols = st.columns(min(len(uploaded), 4))
    for i, f in enumerate(uploaded):
        with preview_cols[i % 4]:
            try:
                img = Image.open(f)
                st.image(img, caption=f.name, use_container_width=True)
            except Exception:
                st.warning(f"Tidak dapat preview: {f.name}")
    st.stop()


# ═══════════════════════════════════════════════════════════════
#  11. INFERENCE & RESULTS
# ═══════════════════════════════════════════════════════════════
st.markdown("<hr class='tz-divider'>", unsafe_allow_html=True)
st.markdown("""
<div class="section-overline">Results</div>
<div class="section-title">Hasil Klasifikasi</div>
""", unsafe_allow_html=True)

batch_records = []

for file_idx, uf in enumerate(uploaded):
    fname = secure_filename(uf.name)
    tmp_path = os.path.join(UPLOAD_DIR, f"tz_{file_idx}_{fname}")
    with open(tmp_path, "wb") as fh:
        fh.write(uf.getbuffer())

    # ── File counter for batch mode
    if len(uploaded) > 1:
        st.markdown(
            f'<div style="font-family:JetBrains Mono,monospace;font-size:0.72rem;'
            f'font-weight:500;color:var(--text-muted);margin:1.2rem 0 0.5rem 0;">'
            f'[{file_idx+1}/{len(uploaded)}] {fname}</div>',
            unsafe_allow_html=True,
        )

    col_img, col_res = st.columns([1, 1], gap="large")

    # ── LEFT COLUMN: Input Image ──
    with col_img:
        st.markdown('<div class="section-overline">Input Image</div>',
                    unsafe_allow_html=True)
        try:
            pil_img = Image.open(tmp_path)
            w, h = pil_img.size
            st.image(pil_img, use_container_width=True)
            st.markdown(
                f'<div class="img-meta">{fname} &nbsp;|&nbsp; {w}&times;{h} px</div>',
                unsafe_allow_html=True,
            )
        except UnidentifiedImageError:
            st.error("File gambar tidak valid atau rusak.")
            continue
        except Exception as e:
            st.error(f"Gagal membuka gambar: {e}")
            continue

    # ── RIGHT COLUMN: Classification Result ──
    with col_res:
        st.markdown('<div class="section-overline">Classification Result</div>',
                    unsafe_allow_html=True)

        with st.spinner("Menganalisis..."):
            try:
                results = model(tmp_path, conf=conf_thresh)
            except Exception as e:
                st.error(f"Inference error: {e}")
                continue

        if not results:
            st.warning("Model tidak mengembalikan hasil.")
            continue

        r = results[0]
        if not (hasattr(r, "probs") and r.probs is not None):
            st.warning("Output bukan klasifikasi. Pastikan model adalah model klasifikasi.")
            continue

        probs_obj  = r.probs
        top1_idx   = int(probs_obj.top1)
        top1_conf  = float(probs_obj.top1conf)
        class_names = r.names if isinstance(r.names, dict) else {}

        try:
            raw = probs_obj.data
            probs_list = raw.cpu().numpy().tolist() if hasattr(raw, "cpu") else list(raw)
            probs_dict = {i: p for i, p in enumerate(probs_list)}
        except Exception:
            probs_dict = {top1_idx: top1_conf}

        top1_name  = class_names.get(top1_idx, f"Class {top1_idx}")
        label_type = classify_label(top1_name)
        low_conf   = top1_conf < conf_thresh
        bar_pct    = top1_conf * 100

        # ── Result Badge
        if low_conf:
            badge_cls, badge_icon = "badge-unknown", "&#9888;"
            badge_text = f"LOW CONFIDENCE &mdash; {top1_name.upper()}"
        elif label_type == "viable":
            badge_cls, badge_icon = "badge-viable", "&#9679;"
            badge_text = top1_name.upper()
        elif label_type == "nonviable":
            badge_cls, badge_icon = "badge-nonviable", "&#9679;"
            badge_text = top1_name.upper()
        else:
            badge_cls, badge_icon = "badge-unknown", "?"
            badge_text = top1_name.upper()

        st.markdown(
            f'<div class="result-badge {badge_cls}">'
            f'<span>{badge_icon}</span> {badge_text}</div>',
            unsafe_allow_html=True,
        )

        # ── Confidence Bar
        effective_type = label_type if not low_conf else "unknown"
        st.markdown(confidence_bar_html(bar_pct, effective_type),
                    unsafe_allow_html=True)

        # ── Stat Cards
        gauge_color = (
            "#059669" if label_type == "viable" and not low_conf else
            "#dc2626" if label_type == "nonviable" and not low_conf else
            "#d97706"
        )
        st.markdown(
            '<div class="stat-row">'
            f'<div class="stat-card">'
            f'  <div class="stat-value" style="color:{gauge_color};">{bar_pct:.1f}%</div>'
            f'  <div class="stat-label">Top Conf.</div>'
            f'</div>'
            f'<div class="stat-card">'
            f'  <div class="stat-value">{top1_idx}</div>'
            f'  <div class="stat-label">Class ID</div>'
            f'</div>'
            f'<div class="stat-card">'
            f'  <div class="stat-value">{len(probs_dict)}</div>'
            f'  <div class="stat-label">Total Kelas</div>'
            f'</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        # ── Gauge Chart
        st.plotly_chart(
            build_gauge(top1_conf, gauge_color),
            use_container_width=True,
            config={"displayModeBar": False},
        )

        batch_records.append({
            "File":           fname,
            "Prediksi":       top1_name,
            "Tipe":           label_type.capitalize(),
            "Confidence (%)": round(bar_pct, 3),
            "Low Conf.":      "Ya" if low_conf else "Tidak",
        })

    # ── PROBABILITY BREAKDOWN (full-width) ──
    st.markdown("""
    <div style="margin-top:1.2rem;">
        <div class="section-overline">Probability Breakdown</div>
    </div>
    """, unsafe_allow_html=True)

    prob_col1, prob_col2 = st.columns([3, 2], gap="large")

    with prob_col1:
        st.plotly_chart(
            build_prob_bar(probs_dict, class_names),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    with prob_col2:
        rows = []
        for idx in sorted(probs_dict, key=lambda x: -probs_dict[x]):
            nm = class_names.get(idx, f"Class {idx}")
            rows.append({
                "Kelas":        nm,
                "Probabilitas": f"{probs_dict[idx]*100:.3f}%",
                "Status":       "✅ Terpilih" if idx == top1_idx else "—",
            })
        df_probs = pd.DataFrame(rows)
        st.dataframe(df_probs, use_container_width=True, hide_index=True,
                     height=min(300, len(rows) * 50 + 50))

    if file_idx < len(uploaded) - 1:
        st.markdown("<hr class='tz-divider'>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  12. BATCH SUMMARY
# ═══════════════════════════════════════════════════════════════
if len(batch_records) > 1:
    st.markdown("<hr class='tz-divider'>", unsafe_allow_html=True)
    st.markdown("""
    <div class="section-overline">Batch Summary</div>
    <div class="section-title">Ringkasan Hasil Batch</div>
    """, unsafe_allow_html=True)

    df_batch = pd.DataFrame(batch_records)
    total    = len(df_batch)
    viable   = (df_batch["Tipe"].str.lower() == "viable").sum()
    nonvia   = (df_batch["Tipe"].str.lower() == "nonviable").sum()
    unknown  = total - viable - nonvia
    avg_conf = df_batch["Confidence (%)"].mean()

    chips_data = [
        (str(total),         "Total Gambar",  "#0f1f15"),
        (str(viable),        "Viable",        "#059669"),
        (str(nonvia),        "Non-Viable",    "#dc2626"),
        (str(unknown),       "Unknown",       "#d97706"),
        (f"{avg_conf:.1f}%", "Avg Confidence", "#0f1f15"),
    ]

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, (val, lbl, clr) in zip([c1, c2, c3, c4, c5], chips_data):
        with col:
            st.markdown(
                f'<div class="stat-card">'
                f'  <div class="stat-value" style="color:{clr};">{val}</div>'
                f'  <div class="stat-label">{lbl}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    col_tbl, col_pie = st.columns([3, 2], gap="large")

    with col_tbl:
        st.dataframe(df_batch, use_container_width=True, hide_index=True)

    with col_pie:
        fig_pie = go.Figure(go.Pie(
            labels=["Viable", "Non-Viable", "Unknown"],
            values=[viable, nonvia, unknown],
            marker_colors=["#059669", "#dc2626", "#d97706"],
            hole=0.58,
            textinfo="label+percent",
            textfont={"color": "#0f1f15", "size": 12, "family": "Inter"},
            hovertemplate="%{label}: %{value} gambar (%{percent})<extra></extra>",
        ))
        fig_pie.update_layout(
            height=260,
            margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            annotations=[dict(
                text=f"<b>{total}</b><br><span style='font-size:10px;'>total</span>",
                x=0.5, y=0.5,
                font_size=18,
                font_color="#0f1f15",
                showarrow=False,
            )],
        )
        st.plotly_chart(fig_pie, use_container_width=True,
                        config={"displayModeBar": False})

    st.markdown("<br>", unsafe_allow_html=True)
    csv_bytes = df_batch.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="\u2b07  Download Hasil Batch (.csv)",
        data=csv_bytes,
        file_name="tz_batch_results.csv",
        mime="text/csv",
        use_container_width=False,
    )


# ═══════════════════════════════════════════════════════════════
#  13. FOOTER
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<hr class='tz-divider'>
<div class="app-footer">
    Tetrazolium Viability Test &nbsp;&middot;&nbsp;
    Seed Analysis Suite &nbsp;&middot;&nbsp;
    &copy; 2025
</div>
""", unsafe_allow_html=True)
