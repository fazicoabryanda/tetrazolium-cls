import streamlit as st
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO
import os
import torch
import pandas as pd
from io import BytesIO
import base64
import plotly.graph_objects as go
from werkzeug.utils import secure_filename

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="TZ Viability Test",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS  –  Clean Green & White Theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Epilogue:wght@300;400;500;600&display=swap');

/* ── Root palette ── */
:root {
    --green:        #3B6D11;
    --green-light:  #EAF3DE;
    --green-mid:    #639922;
    --green-border: #C0DD97;
    --green-dark:   #27500A;
    --red:          #A32D2D;
    --red-light:    #FCEBEB;
    --red-mid:      #E24B4A;
    --red-border:   #F7C1C1;
    --text:         #1a1a1a;
    --muted:        #6b7280;
    --border:       #e5e7eb;
    --surface:      #f9faf8;
    --white:        #ffffff;
    --mono:         'IBM Plex Mono', monospace;
    --body:         'Epilogue', sans-serif;
}

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'Epilogue', sans-serif !important;
    background-color: var(--surface) !important;
    color: var(--text) !important;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ── Topbar ── */
.tz-topbar {
    background: var(--white);
    border-bottom: 1px solid var(--border);
    padding: 12px 28px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0;
}
.tz-brand-name {
    font-family: var(--mono);
    font-size: 13px;
    font-weight: 500;
    color: var(--green);
    letter-spacing: 0.04em;
}
.tz-brand-sub {
    font-size: 11px;
    color: var(--muted);
    margin-top: 2px;
}
.tz-brand-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    background: var(--green-mid);
    border-radius: 50%;
    margin-right: 8px;
    vertical-align: middle;
}

/* ── Status pill ── */
.tz-status-ok {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--green-light);
    border: 1px solid var(--green-border);
    border-radius: 99px;
    padding: 4px 14px;
    font-family: var(--mono);
    font-size: 11px;
    color: var(--green);
}
.tz-status-err {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--red-light);
    border: 1px solid var(--red-border);
    border-radius: 99px;
    padding: 4px 14px;
    font-family: var(--mono);
    font-size: 11px;
    color: var(--red);
}
.tz-pulse {
    display: inline-block;
    width: 6px;
    height: 6px;
    background: var(--green-mid);
    border-radius: 50%;
    animation: tz-pulse 2s infinite;
}
@keyframes tz-pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}

/* ── Section labels ── */
.tz-section-tag {
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 0.14em;
    color: var(--green-mid);
    text-transform: uppercase;
    margin-bottom: 10px;
}

/* ── Cards ── */
.tz-card {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px 18px;
    margin-bottom: 14px;
}
.tz-card-accent {
    border-left: 3px solid var(--green-mid);
}

/* ── Result badge ── */
.tz-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 7px 16px;
    border-radius: 6px;
    font-family: var(--mono);
    font-size: 13px;
    font-weight: 500;
    margin-bottom: 12px;
}
.tz-badge-viable {
    background: var(--green-light);
    border: 1px solid var(--green-border);
    color: var(--green);
}
.tz-badge-nonviable {
    background: var(--red-light);
    border: 1px solid var(--red-border);
    color: var(--red);
}
.tz-badge-unknown {
    background: #f3f4f6;
    border: 1px solid var(--border);
    color: var(--muted);
}
.tz-badge-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: currentColor;
    flex-shrink: 0;
}

/* ── Confidence bar ── */
.tz-conf-wrap { margin: 10px 0; }
.tz-conf-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
}
.tz-conf-label { font-size: 12px; color: var(--muted); }
.tz-conf-track {
    height: 6px;
    background: #f1f5f0;
    border-radius: 99px;
    overflow: hidden;
    border: 1px solid var(--border);
}
.tz-conf-fill-viable {
    height: 100%;
    border-radius: 99px;
    background: var(--green-mid);
    transition: width 0.5s ease;
}
.tz-conf-fill-nonviable {
    height: 100%;
    border-radius: 99px;
    background: var(--red-mid);
    transition: width 0.5s ease;
}
.tz-conf-fill-unknown {
    height: 100%;
    border-radius: 99px;
    background: #9ca3af;
}
.tz-conf-footer {
    margin-top: 5px;
    font-family: var(--mono);
    font-size: 10px;
    color: var(--muted);
}

/* ── Stat chips ── */
.tz-stat-row {
    display: flex;
    gap: 10px;
    margin: 12px 0;
    flex-wrap: wrap;
}
.tz-stat-chip {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 14px;
    text-align: center;
    flex: 1;
    min-width: 80px;
}
.tz-stat-val {
    font-family: var(--mono);
    font-size: 18px;
    font-weight: 500;
    color: var(--text);
    line-height: 1;
}
.tz-stat-lbl {
    font-size: 10px;
    color: var(--muted);
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* ── Info note ── */
.tz-note {
    background: var(--green-light);
    border-left: 3px solid var(--green-mid);
    border-radius: 0 6px 6px 0;
    padding: 10px 14px;
    font-size: 12px;
    color: var(--green-dark);
    line-height: 1.6;
    margin: 10px 0;
}
.tz-note-warn {
    background: var(--red-light);
    border-left: 3px solid var(--red-mid);
    border-radius: 0 6px 6px 0;
    padding: 10px 14px;
    font-size: 12px;
    color: var(--red);
    line-height: 1.6;
    margin: 10px 0;
}

/* ── Divider ── */
.tz-divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 18px 0;
}

/* ── File info ── */
.tz-file-info {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--muted);
    margin-top: 5px;
    display: flex;
    gap: 10px;
}

/* ── Sidebar overrides ── */
[data-testid="stSidebar"] {
    background: var(--white) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stSidebar"] .tz-muted { color: var(--muted) !important; }

/* ── Widget overrides ── */
.stFileUploader > div {
    background: var(--surface) !important;
    border: 1.5px dashed var(--green-border) !important;
    border-radius: 8px !important;
}
.stButton > button {
    background: var(--green-mid) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: var(--mono) !important;
    font-size: 12px !important;
    letter-spacing: 0.04em !important;
    padding: 7px 18px !important;
    transition: background 0.2s !important;
}
.stButton > button:hover { background: var(--green) !important; }

.stSlider > div > div > div { background: var(--green-mid) !important; }

div[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    overflow: hidden;
    font-size: 12px !important;
}

/* ── Batch summary table ── */
.tz-summary-chip {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px 16px;
    text-align: center;
}
.tz-summary-val {
    font-family: var(--mono);
    font-size: 22px;
    font-weight: 500;
    line-height: 1;
}
.tz-summary-lbl {
    font-size: 11px;
    color: var(--muted);
    margin-top: 5px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--surface); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
MODEL_PATH = "tetrazolium_model.pt"
UPLOAD_DIR = "tz_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ─────────────────────────────────────────────
#  MODEL LOADER
# ─────────────────────────────────────────────
@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        return None, f"Model file tidak ditemukan: `{path}`"
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


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def classify_label(name: str) -> str:
    n = name.lower()
    if "nonviable" in n or "non_viable" in n or "non-viable" in n:
        return "nonviable"
    if "viable" in n:
        return "viable"
    return "unknown"


def build_gauge(value: float, color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(value * 100, 2),
        number={
            "suffix": "%",
            "font": {"size": 26, "color": "#1a1a1a", "family": "IBM Plex Mono"},
        },
        gauge={
            "axis": {
                "range": [0, 100],
                "tickcolor": "#9ca3af",
                "tickfont": {"color": "#9ca3af", "size": 10},
                "nticks": 6,
            },
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "#f9faf8",
            "bordercolor": "#e5e7eb",
            "borderwidth": 1,
            "steps": [
                {"range": [0, 50],  "color": "#f3f4f6"},
                {"range": [50, 75], "color": "#f9faf8"},
                {"range": [75, 100], "color": "#f9faf8"},
            ],
            "threshold": {
                "line": {"color": color, "width": 2},
                "thickness": 0.8,
                "value": value * 100,
            },
        },
        title={
            "text": "Confidence Score",
            "font": {"color": "#6b7280", "size": 11, "family": "Epilogue"},
        },
    ))
    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=50, b=5),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#1a1a1a",
    )
    return fig


def build_prob_bar(probs: dict, class_names: dict) -> go.Figure:
    labels, values, colors = [], [], []
    color_map = {
        "viable":    "#639922",
        "nonviable": "#E24B4A",
        "unknown":   "#9ca3af",
    }
    for idx, prob in sorted(probs.items(), key=lambda x: -x[1]):
        name = class_names.get(idx, f"Class {idx}")
        labels.append(name)
        values.append(round(prob * 100, 3))
        colors.append(color_map.get(classify_label(name), "#9ca3af"))

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=colors,
        text=[f"{v:.2f}%" for v in values],
        textposition="outside",
        textfont={"color": "#6b7280", "size": 11, "family": "IBM Plex Mono"},
        hovertemplate="%{y}: %{x:.3f}%<extra></extra>",
    ))
    fig.update_layout(
        height=max(160, len(labels) * 52 + 40),
        margin=dict(l=10, r=80, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            range=[0, 118],
            showgrid=True,
            gridcolor="#f3f4f6",
            tickfont={"color": "#9ca3af", "size": 10},
            zeroline=False,
        ),
        yaxis=dict(tickfont={"color": "#1a1a1a", "size": 12}),
        bargap=0.4,
    )
    return fig


def build_pie(viable: int, nonvia: int, unknown: int, total: int) -> go.Figure:
    pie_labels = ["Viable", "Non-Viable", "Unknown"]
    pie_values = [viable, nonvia, unknown]
    pie_colors = ["#639922", "#E24B4A", "#9ca3af"]
    fig = go.Figure(go.Pie(
        labels=pie_labels, values=pie_values,
        marker_colors=pie_colors,
        hole=0.58,
        textinfo="label+percent",
        textfont={"color": "#1a1a1a", "size": 11, "family": "Epilogue"},
        hovertemplate="%{label}: %{value} gambar (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        height=260,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        annotations=[dict(
            text=f"<b>{total}</b><br><span style='font-size:10px'>total</span>",
            x=0.5, y=0.5,
            font_size=18,
            font_color="#1a1a1a",
            showarrow=False,
        )],
    )
    return fig


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:16px 0 10px 0;">
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:12px;">
            <div style="width:8px;height:8px;background:#639922;border-radius:50%;"></div>
            <div>
                <div style="font-family:'IBM Plex Mono',monospace;font-size:12px;
                            font-weight:500;color:#3B6D11;letter-spacing:0.04em;">
                    TZ Viability Test
                </div>
                <div style="font-size:10px;color:#6b7280;margin-top:1px;">
                    Seed Analysis Suite · v2.0
                </div>
            </div>
        </div>
    </div>
    <hr style="border:none;border-top:1px solid #e5e7eb;margin:0 0 16px 0;">
    """, unsafe_allow_html=True)

    st.markdown('<div class="tz-section-tag">Model</div>', unsafe_allow_html=True)
    model_path_input = st.text_input(
        "Path model (.pt)", value=MODEL_PATH,
        help="Path ke file model klasifikasi Tetrazolium YOLOv11",
    )

    st.markdown('<div class="tz-section-tag" style="margin-top:16px;">Threshold</div>',
                unsafe_allow_html=True)
    conf_thresh = st.slider(
        "Confidence threshold", min_value=0.01, max_value=0.99,
        value=0.25, step=0.01,
        help="Hasil di bawah threshold akan ditandai sebagai low confidence.",
    )
    st.caption(f"Threshold aktif: `{conf_thresh:.2f}`")

    st.markdown('<div class="tz-section-tag" style="margin-top:16px;">Mode</div>',
                unsafe_allow_html=True)
    batch_mode = st.toggle(
        "Batch mode (multi-gambar)", value=False,
        help="Upload beberapa gambar sekaligus untuk diklasifikasi batch.",
    )

    st.markdown("""
    <hr style="border:none;border-top:1px solid #e5e7eb;margin:20px 0 14px 0;">
    <div style="font-size:11px;color:#6b7280;line-height:1.7;">
        <b style="color:#1a1a1a;">Tentang Uji Tetrazolium</b><br>
        Uji viabilitas benih menggunakan garam TTC
        (2,3,5-triphenyltetrazolium chloride).<br><br>
        <span style="color:#639922;font-weight:500;">● Viable</span>
        — jaringan aktif, berwarna merah<br>
        <span style="color:#E24B4A;font-weight:500;">● Non-Viable</span>
        — jaringan mati, tidak berwarna
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────────
model, model_err = load_model(model_path_input)

# ─────────────────────────────────────────────
#  TOPBAR
# ─────────────────────────────────────────────
if model_err:
    status_html = (
        '<span class="tz-status-err">'
        '<span style="display:inline-block;width:6px;height:6px;'
        'background:#E24B4A;border-radius:50%;"></span>'
        'Model error</span>'
    )
else:
    class_names = getattr(model, "names", {})
    n_classes = len(class_names)
    status_html = (
        f'<span class="tz-status-ok">'
        f'<span class="tz-pulse"></span>'
        f'Model loaded &nbsp;·&nbsp; {n_classes} kelas</span>'
    )

st.markdown(f"""
<div class="tz-topbar">
    <div style="display:flex;align-items:center;gap:8px;">
        <span class="tz-brand-dot"></span>
        <div>
            <div class="tz-brand-name">TZ Viability Test</div>
            <div class="tz-brand-sub">Seed Science &amp; Technology · Deep Learning Classifier</div>
        </div>
    </div>
    {status_html}
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  MODEL ERROR STATE
# ─────────────────────────────────────────────
if model_err:
    st.markdown(f"""
    <div style="padding:28px 32px;">
        <div class="tz-card" style="border-left:3px solid #E24B4A;max-width:640px;">
            <div class="tz-section-tag" style="color:#E24B4A;">Model Error</div>
            <code style="color:#A32D2D;font-size:13px;font-family:'IBM Plex Mono',monospace;">
                {model_err}
            </code>
            <div class="tz-note-warn" style="margin-top:12px;">
                Pastikan file <code>tetrazolium_model.pt</code> berada di direktori yang sama
                dengan script ini, atau ubah path model di sidebar.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────
#  MAIN CONTENT PADDING WRAPPER
# ─────────────────────────────────────────────
st.markdown('<div style="padding:24px 32px 40px 32px;">', unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  UPLOADER
# ─────────────────────────────────────────────
st.markdown('<div class="tz-section-tag">Input</div>', unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Upload gambar hasil uji Tetrazolium",
    type=["png", "jpg", "jpeg", "bmp", "tiff"],
    accept_multiple_files=batch_mode,
    key="tz_uploader",
)

if not batch_mode and uploaded is not None:
    uploaded = [uploaded]

if not uploaded:
    st.markdown("""
    <div class="tz-note">
        📌 Upload gambar benih setelah proses pewarnaan tetrazolium.
        Format: PNG, JPG, JPEG, BMP, TIFF.
        Aktifkan <strong>Batch Mode</strong> di sidebar untuk mengklasifikasi banyak gambar sekaligus.
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────
#  CLASSIFY BUTTON
# ─────────────────────────────────────────────
run_btn = st.button(
    f"🔬  Klasifikasi {'Semua Gambar' if len(uploaded) > 1 else 'Gambar'}",
    use_container_width=False,
    type="primary",
)

if not run_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    cols = st.columns(min(len(uploaded), 4))
    for i, f in enumerate(uploaded):
        with cols[i % 4]:
            try:
                img = Image.open(f)
                st.image(img, caption=f.name, use_container_width=True)
            except Exception:
                st.warning(f"Tidak dapat preview: {f.name}")
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────
#  INFERENCE LOOP
# ─────────────────────────────────────────────
st.markdown('<hr class="tz-divider">', unsafe_allow_html=True)
st.markdown('<div class="tz-section-tag">Hasil Klasifikasi</div>', unsafe_allow_html=True)

batch_records = []

for file_idx, uf in enumerate(uploaded):
    fname = secure_filename(uf.name)
    tmp_path = os.path.join(UPLOAD_DIR, f"tz_{file_idx}_{fname}")
    with open(tmp_path, "wb") as fh:
        fh.write(uf.getbuffer())

    if len(uploaded) > 1:
        st.markdown(
            f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:11px;'
            f'color:#6b7280;margin:16px 0 6px 0;">'
            f'[{file_idx + 1}/{len(uploaded)}] &nbsp;{fname}</div>',
            unsafe_allow_html=True,
        )

    col_img, col_res = st.columns([1, 1], gap="large")

    # ── Left: image preview ──────────────────
    with col_img:
        st.markdown('<div class="tz-section-tag">Input Image</div>', unsafe_allow_html=True)
        try:
            pil_img = Image.open(tmp_path)
            w, h = pil_img.size
            st.image(pil_img, use_container_width=True)
            st.markdown(
                f'<div class="tz-file-info"><span>{fname}</span>'
                f'<span>{w}×{h} px</span></div>',
                unsafe_allow_html=True,
            )
        except UnidentifiedImageError:
            st.error("File gambar tidak valid atau rusak.")
            continue
        except Exception as e:
            st.error(f"Gagal membuka gambar: {e}")
            continue

    # ── Right: inference + results ───────────
    with col_res:
        st.markdown('<div class="tz-section-tag">Classification Result</div>',
                    unsafe_allow_html=True)
        with st.spinner("Menganalisis gambar..."):
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

        probs_obj   = r.probs
        top1_idx    = int(probs_obj.top1)
        top1_conf   = float(probs_obj.top1conf)
        class_names = r.names if isinstance(r.names, dict) else {}

        # Build probability dict
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

        # ── Determine badge style
        if low_conf:
            badge_css  = "tz-badge-unknown"
            badge_text = f"LOW CONFIDENCE — {top1_name}"
            conf_color = "#9ca3af"
            bar_css    = "tz-conf-fill-unknown"
            gauge_color = "#9ca3af"
        elif label_type == "viable":
            badge_css  = "tz-badge-viable"
            badge_text = top1_name.upper()
            conf_color = "#639922"
            bar_css    = "tz-conf-fill-viable"
            gauge_color = "#639922"
        elif label_type == "nonviable":
            badge_css  = "tz-badge-nonviable"
            badge_text = top1_name.upper()
            conf_color = "#E24B4A"
            bar_css    = "tz-conf-fill-nonviable"
            gauge_color = "#E24B4A"
        else:
            badge_css  = "tz-badge-unknown"
            badge_text = top1_name.upper()
            conf_color = "#9ca3af"
            bar_css    = "tz-conf-fill-unknown"
            gauge_color = "#9ca3af"

        # ── Badge
        st.markdown(
            f'<div class="tz-badge {badge_css}">'
            f'<span class="tz-badge-dot"></span>{badge_text}</div>',
            unsafe_allow_html=True,
        )

        # ── Confidence bar (fixed: no broken HTML)
        low_conf_note = (
            f'&nbsp;<span style="color:#9ca3af;font-size:10px;">'
            f'(threshold: {conf_thresh * 100:.0f}%)</span>'
            if low_conf else ""
        )
        st.markdown(
            f'<div class="tz-conf-wrap">'
            f'  <div class="tz-conf-header">'
            f'    <span class="tz-conf-label">Confidence Score</span>'
            f'    <span style="font-family:\'IBM Plex Mono\',monospace;font-size:13px;'
            f'font-weight:500;color:{conf_color};">{bar_pct:.2f}%{low_conf_note}</span>'
            f'  </div>'
            f'  <div class="tz-conf-track">'
            f'    <div class="{bar_css}" style="width:{bar_pct:.2f}%;"></div>'
            f'  </div>'
            f'  <div class="tz-conf-footer">Threshold: {conf_thresh * 100:.0f}%</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ── Stat chips
        st.markdown(
            f'<div class="tz-stat-row">'
            f'  <div class="tz-stat-chip">'
            f'    <div class="tz-stat-val" style="color:{conf_color};">{bar_pct:.1f}%</div>'
            f'    <div class="tz-stat-lbl">Top Conf.</div>'
            f'  </div>'
            f'  <div class="tz-stat-chip">'
            f'    <div class="tz-stat-val">{top1_idx}</div>'
            f'    <div class="tz-stat-lbl">Class ID</div>'
            f'  </div>'
            f'  <div class="tz-stat-chip">'
            f'    <div class="tz-stat-val">{len(probs_dict)}</div>'
            f'    <div class="tz-stat-lbl">Total Kelas</div>'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ── Gauge
        st.plotly_chart(
            build_gauge(top1_conf, gauge_color),
            use_container_width=True,
            config={"displayModeBar": False},
        )

        # Record for batch summary
        batch_records.append({
            "File":            fname,
            "Prediksi":        top1_name,
            "Tipe":            label_type.capitalize(),
            "Confidence (%)":  round(bar_pct, 3),
            "Low Confidence":  "Ya" if low_conf else "Tidak",
        })

    # ── Probability Breakdown (full width) ──
    st.markdown('<hr class="tz-divider">', unsafe_allow_html=True)
    st.markdown('<div class="tz-section-tag">Probability Breakdown</div>',
                unsafe_allow_html=True)

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
                "Probabilitas": f"{probs_dict[idx] * 100:.3f}%",
                "Status":       "✅ Terpilih" if idx == top1_idx else "—",
            })
        df_probs = pd.DataFrame(rows)
        st.dataframe(df_probs, use_container_width=True, hide_index=True,
                     height=min(300, len(rows) * 50 + 50))

    if file_idx < len(uploaded) - 1:
        st.markdown('<hr class="tz-divider">', unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  BATCH SUMMARY  (only when >1 image)
# ─────────────────────────────────────────────
if len(batch_records) > 1:
    st.markdown('<hr class="tz-divider">', unsafe_allow_html=True)
    st.markdown('<div class="tz-section-tag">Batch Summary</div>', unsafe_allow_html=True)

    df_batch = pd.DataFrame(batch_records)
    total    = len(df_batch)
    viable   = (df_batch["Tipe"].str.lower() == "viable").sum()
    nonvia   = (df_batch["Tipe"].str.lower() == "nonviable").sum()
    unknown  = total - viable - nonvia
    avg_conf = df_batch["Confidence (%)"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    chips = [
        (str(total),           "Total",         "#1a1a1a"),
        (str(viable),          "Viable",         "#639922"),
        (str(nonvia),          "Non-Viable",     "#E24B4A"),
        (str(unknown),         "Unknown",        "#9ca3af"),
        (f"{avg_conf:.1f}%",  "Avg Confidence", "#1a1a1a"),
    ]
    for col, (val, lbl, clr) in zip([c1, c2, c3, c4, c5], chips):
        with col:
            st.markdown(
                f'<div class="tz-summary-chip">'
                f'  <div class="tz-summary-val" style="color:{clr};">{val}</div>'
                f'  <div class="tz-summary-lbl">{lbl}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    col_tbl, col_pie = st.columns([3, 2], gap="large")
    with col_tbl:
        st.dataframe(df_batch, use_container_width=True, hide_index=True)
    with col_pie:
        st.plotly_chart(
            build_pie(viable, nonvia, unknown, total),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    st.markdown("<br>", unsafe_allow_html=True)
    csv_bytes = df_batch.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇  Download Hasil Batch (.csv)",
        data=csv_bytes,
        file_name="tz_batch_results.csv",
        mime="text/csv",
    )

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<hr class="tz-divider">
<div style="text-align:center;font-size:11px;color:#9ca3af;padding:4px 0 8px 0;
            font-family:'IBM Plex Mono',monospace;letter-spacing:0.04em;">
    Tetrazolium Viability Test &nbsp;·&nbsp; Seed Analysis Suite &nbsp;·&nbsp; ©2025
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
