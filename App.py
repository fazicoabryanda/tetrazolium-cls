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
#  CUSTOM CSS  –  Clean Green & White Lab Theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

:root {
    --bg:           #f5f7f4;
    --surface:      #ffffff;
    --surface2:     #eef3ec;
    --border:       #d4e0d0;
    --accent:       #2d7a4f;
    --accent-light: #e8f5ee;
    --accent-dark:  #1a5235;
    --viable:       #2d7a4f;
    --nonviable:    #c0392b;
    --warn:         #e67e22;
    --text:         #1a2e1e;
    --muted:        #6b8a70;
}

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 1.5rem 2.5rem 4rem 2.5rem !important;
    max-width: 1440px;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: var(--accent-dark) !important;
    border-right: none !important;
}
[data-testid="stSidebar"] * { color: #d4ead9 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] strong { color: #ffffff !important; }
[data-testid="stSidebar"] .stTextInput input {
    background: rgba(255,255,255,0.1) !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    color: #ffffff !important;
    border-radius: 6px !important;
}
[data-testid="stSidebar"] .stSlider > div > div > div {
    background: #5eb87c !important;
}

/* ── HERO ── */
.hero {
    background: linear-gradient(135deg, var(--accent-dark) 0%, var(--accent) 60%, #3d9e65 100%);
    border-radius: 16px;
    padding: 2.8rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(45,122,79,0.25);
}
.hero::after {
    content: '🌿';
    position: absolute;
    right: 3rem;
    top: 50%;
    transform: translateY(-50%);
    font-size: 6rem;
    opacity: 0.15;
    pointer-events: none;
}
.hero-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.22em;
    color: #a8d8b5;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}
.hero-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 2.4rem;
    font-weight: 700;
    color: #ffffff;
    line-height: 1.2;
    margin: 0 0 0.8rem 0;
}
.hero-title span {
    color: #a8d8b5;
    font-weight: 300;
}
.hero-subtitle {
    font-size: 0.92rem;
    color: rgba(255,255,255,0.75);
    max-width: 600px;
    line-height: 1.7;
}

/* ── STATUS BANNER ── */
.status-ok {
    background: var(--accent-light);
    border: 1.5px solid #b2d9be;
    border-radius: 10px;
    padding: 0.85rem 1.4rem;
    display: flex;
    align-items: center;
    gap: 0.7rem;
    font-size: 0.88rem;
    margin-bottom: 1.5rem;
}
.status-err {
    background: #fdf2f0;
    border: 1.5px solid #f1b4ab;
    border-radius: 10px;
    padding: 0.85rem 1.4rem;
    font-size: 0.88rem;
    margin-bottom: 1.5rem;
}
.dot-ok  { width:10px;height:10px;border-radius:50%;background:var(--viable);flex-shrink:0; }
.dot-err { width:10px;height:10px;border-radius:50%;background:var(--nonviable);flex-shrink:0; }

/* ── SECTION LABELS ── */
.sec-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.2em;
    color: var(--muted);
    text-transform: uppercase;
    margin-bottom: 0.2rem;
}
.sec-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 1rem;
}

/* ── RESULT BADGE ── */
.result-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.65rem 1.4rem;
    border-radius: 9px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    margin: 0.5rem 0 0.8rem 0;
}
.badge-viable {
    background: var(--accent-light);
    border: 2px solid var(--viable);
    color: var(--viable);
}
.badge-nonviable {
    background: #fdf2f0;
    border: 2px solid var(--nonviable);
    color: var(--nonviable);
}
.badge-unknown {
    background: #fef8ec;
    border: 2px solid var(--warn);
    color: var(--warn);
}

/* ── CONFIDENCE BAR ── */
.conf-wrap { margin: 0.8rem 0 1.2rem 0; }
.conf-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
}
.conf-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
.conf-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.88rem;
    font-weight: 600;
}
.conf-track {
    background: var(--surface2);
    border-radius: 99px;
    height: 12px;
    width: 100%;
    overflow: hidden;
    border: 1px solid var(--border);
}
.conf-fill-viable {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, #2d7a4f, #5eb87c);
}
.conf-fill-nonviable {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, #a93226, #c0392b);
}
.conf-fill-unknown {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, #c87d2b, #e67e22);
}

/* ── STAT CHIPS ── */
.stat-row { display:flex;gap:0.75rem;flex-wrap:wrap;margin:1rem 0; }
.stat-chip {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.8rem 1.1rem;
    text-align: center;
    min-width: 100px;
    flex: 1;
}
.stat-val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.35rem;
    font-weight: 600;
    color: var(--text);
}
.stat-lbl {
    font-size: 0.68rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.2rem;
}

/* ── INFO BOX ── */
.info-box {
    background: var(--accent-light);
    border-left: 3px solid var(--accent);
    border-radius: 0 8px 8px 0;
    padding: 0.85rem 1.1rem;
    margin: 0.8rem 0;
    font-size: 0.84rem;
    color: var(--accent-dark);
    line-height: 1.65;
}

/* ── DIVIDER ── */
.tz-divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 1.8rem 0;
}

/* ── IMAGE FRAME ── */
.img-meta {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: var(--muted);
    margin-top: 0.4rem;
}

/* ── FILE UPLOADER ── */
.stFileUploader > div {
    background: var(--surface) !important;
    border: 2px dashed var(--border) !important;
    border-radius: 12px !important;
}
.stFileUploader > div:hover {
    border-color: var(--accent) !important;
}

/* ── BUTTON ── */
.stButton > button {
    background: var(--accent) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 9px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    padding: 0.65rem 2rem !important;
    box-shadow: 0 4px 16px rgba(45,122,79,0.25) !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: var(--accent-dark) !important;
    box-shadow: 0 6px 20px rgba(45,122,79,0.35) !important;
}

div[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    overflow: hidden;
}

::-webkit-scrollbar { width:6px;height:6px; }
::-webkit-scrollbar-track { background:var(--surface2); }
::-webkit-scrollbar-thumb { background:var(--border);border-radius:3px; }
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


def confidence_bar_html(pct: float, label_type: str) -> str:
    """Render confidence bar safely — no raw string injection from model output."""
    if label_type == "viable":
        fill_class = "conf-fill-viable"
        color = "#2d7a4f"
    elif label_type == "nonviable":
        fill_class = "conf-fill-nonviable"
        color = "#c0392b"
    else:
        fill_class = "conf-fill-unknown"
        color = "#e67e22"

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
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(value * 100, 2),
        number={
            "suffix": "%",
            "font": {"size": 30, "color": "#1a2e1e", "family": "IBM Plex Mono"},
        },
        gauge={
            "axis": {
                "range": [0, 100],
                "tickcolor": "#6b8a70",
                "tickfont": {"color": "#6b8a70", "size": 10},
            },
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "#eef3ec",
            "bordercolor": "#d4e0d0",
            "borderwidth": 1,
            "steps": [
                {"range": [0, 50],  "color": "#f5f7f4"},
                {"range": [50, 75], "color": "#e8f5ee" if color == "#2d7a4f" else "#fdf2f0"},
                {"range": [75, 100], "color": "#d4eddb" if color == "#2d7a4f" else "#f9d6d2"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.85,
                "value": value * 100,
            },
        },
        title={
            "text": "Confidence Level",
            "font": {"color": "#6b8a70", "size": 11, "family": "Plus Jakarta Sans"},
        },
    ))
    fig.update_layout(
        height=210,
        margin=dict(l=20, r=20, t=40, b=5),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#1a2e1e",
    )
    return fig


def build_prob_bar(probs: dict, class_names: dict) -> go.Figure:
    labels, values, colors = [], [], []
    color_map = {
        "viable":    "#2d7a4f",
        "nonviable": "#c0392b",
        "unknown":   "#e67e22",
    }
    for idx, prob in sorted(probs.items(), key=lambda x: -x[1]):
        name = class_names.get(idx, f"Class {idx}")
        labels.append(name)
        values.append(round(prob * 100, 3))
        colors.append(color_map.get(classify_label(name), "#6b8a70"))

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=colors,
        marker_line_color="rgba(0,0,0,0.05)",
        marker_line_width=1,
        text=[f"{v:.2f}%" for v in values],
        textposition="outside",
        textfont={"color": "#1a2e1e", "size": 12, "family": "IBM Plex Mono"},
        hovertemplate="%{y}: %{x:.3f}%<extra></extra>",
    ))
    fig.update_layout(
        height=max(160, len(labels) * 60),
        margin=dict(l=10, r=80, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            range=[0, 120],
            showgrid=True,
            gridcolor="#d4e0d0",
            tickfont={"color": "#6b8a70"},
            zeroline=False,
        ),
        yaxis=dict(tickfont={"color": "#1a2e1e", "size": 13}),
        bargap=0.4,
    )
    return fig


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:1.2rem 0 0.4rem 0;">
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.6rem;
                    letter-spacing:0.2em;color:#a8d8b5;text-transform:uppercase;
                    margin-bottom:0.4rem;">Seed Science</div>
        <div style="font-size:1.2rem;font-weight:700;color:#ffffff;">
            TZ Test Config
        </div>
    </div>
    <hr style="border-color:rgba(255,255,255,0.15);margin:0.8rem 0 1.2rem 0;">
    """, unsafe_allow_html=True)

    st.markdown("**Model Path**")
    model_path_input = st.text_input(
        "Model path (.pt)", value=MODEL_PATH,
        help="Path ke file model klasifikasi Tetrazolium YOLOv11",
        label_visibility="collapsed",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Confidence Threshold**")
    conf_thresh = st.slider(
        "conf", min_value=0.01, max_value=0.99,
        value=0.25, step=0.01,
        label_visibility="collapsed",
        help="Hasil di bawah threshold ini akan ditandai sebagai 'low confidence'.",
    )
    st.caption(f"Threshold aktif: `{conf_thresh:.2f}`")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Batch Mode**")
    batch_mode = st.toggle(
        "Aktifkan multi-gambar", value=False,
        help="Upload beberapa gambar sekaligus untuk diklasifikasi batch.",
    )

    st.markdown("""
    <hr style="border-color:rgba(255,255,255,0.15);margin:1.5rem 0 1rem 0;">
    <div style="font-size:0.75rem;color:#a8d8b5;line-height:1.8;">
        <b style="color:#ffffff;font-size:0.82rem;">Tetrazolium (TZ) Test</b><br><br>
        Uji viabilitas benih menggunakan garam tetrazolium
        (2,3,5-triphenyltetrazolium chloride). Benih <em>viable</em> berwarna merah
        karena aktivitas enzim dehidrogenase.<br><br>
        <span style="color:#5eb87c;">&#9679; Viable</span> — jaringan aktif, berwarna merah<br>
        <span style="color:#e8817a;">&#9679; Non-Viable</span> — jaringan mati, tidak berwarna
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────────
model, model_err = load_model(model_path_input)

# ─────────────────────────────────────────────
#  HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-label">&#127807; Seed Science &amp; Technology Lab</div>
    <h1 class="hero-title">Tetrazolium <span>Viability</span> Test</h1>
    <p class="hero-subtitle">
        Klasifikasi viabilitas benih secara otomatis menggunakan deep learning YOLOv11.
        Upload gambar hasil pewarnaan TZ untuk menentukan apakah benih
        <strong>viable</strong> atau <strong>non-viable</strong> beserta skor kepercayaannya.
    </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  MODEL STATUS
# ─────────────────────────────────────────────
if model_err:
    st.markdown(
        '<div class="status-err">'
        '<span style="display:inline-flex;align-items:center;gap:0.5rem;">'
        '<span class="dot-err"></span>'
        f'<strong style="color:#c0392b;">Model Error:</strong>&nbsp;'
        f'<code style="color:#c0392b;">{model_err}</code>'
        '</span><br>'
        '<span style="color:#7f8c8d;font-size:0.82rem;margin-top:0.4rem;display:block;">'
        'Pastikan file <code>tetrazolium_model.pt</code> berada di direktori yang sama '
        'dengan script ini, atau ubah path model di sidebar.'
        '</span></div>',
        unsafe_allow_html=True,
    )
else:
    class_names = getattr(model, "names", {})
    class_list_html = " &nbsp;&middot;&nbsp; ".join(
        f'<code style="background:#eef3ec;padding:2px 6px;border-radius:4px;'
        f'font-size:0.8rem;color:#1a5235;">{v}</code>'
        for v in class_names.values()
    ) if class_names else "&mdash;"
    st.markdown(
        '<div class="status-ok">'
        '<span class="dot-ok"></span>'
        '<span style="font-weight:600;color:#1a5235;">Model berhasil dimuat</span>'
        '<span style="color:#d4e0d0;margin:0 0.5rem;">|</span>'
        f'<span style="font-size:0.82rem;color:#6b8a70;">Kelas: {class_list_html}</span>'
        '</div>',
        unsafe_allow_html=True,
    )

st.markdown("<hr class='tz-divider'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  UPLOADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="sec-label">Input</div>
<div class="sec-title">Upload Gambar Hasil Uji TZ</div>
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

# ─────────────────────────────────────────────
#  CLASSIFY BUTTON
# ─────────────────────────────────────────────
run_btn = st.button(
    f"\U0001f52c  Klasifikasi {'Semua Gambar' if len(uploaded) > 1 else 'Gambar'}",
    use_container_width=True,
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
    st.stop()

# ─────────────────────────────────────────────
#  INFERENCE LOOP
# ─────────────────────────────────────────────
st.markdown("<hr class='tz-divider'>", unsafe_allow_html=True)
st.markdown("""
<div class="sec-label">Results</div>
<div class="sec-title">Hasil Klasifikasi</div>
""", unsafe_allow_html=True)

batch_records = []

for file_idx, uf in enumerate(uploaded):
    fname = secure_filename(uf.name)
    tmp_path = os.path.join(UPLOAD_DIR, f"tz_{file_idx}_{fname}")
    with open(tmp_path, "wb") as fh:
        fh.write(uf.getbuffer())

    if len(uploaded) > 1:
        st.markdown(
            f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:0.75rem;'
            f'color:#6b8a70;margin:1.2rem 0 0.5rem 0;">'
            f'[{file_idx+1}/{len(uploaded)}] {fname}</div>',
            unsafe_allow_html=True,
        )

    col_img, col_res = st.columns([1, 1], gap="large")

    # ── Left: image
    with col_img:
        st.markdown('<div class="sec-label">Input Image</div>', unsafe_allow_html=True)
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

    # ── Right: results
    with col_res:
        st.markdown('<div class="sec-label">Classification Result</div>',
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

        probs_obj   = r.probs
        top1_idx    = int(probs_obj.top1)
        top1_conf   = float(probs_obj.top1conf)
        class_names = r.names if isinstance(r.names, dict) else {}

        try:
            raw = probs_obj.data
            probs_list = raw.cpu().numpy().tolist() if hasattr(raw, "cpu") else list(raw)
            probs_dict = {i: p for i, p in enumerate(probs_list)}
        except Exception:
            probs_dict = {top1_idx: top1_conf}

        top1_name   = class_names.get(top1_idx, f"Class {top1_idx}")
        label_type  = classify_label(top1_name)
        low_conf    = top1_conf < conf_thresh
        bar_pct     = top1_conf * 100

        # Badge
        if low_conf:
            badge_cls  = "badge-unknown"
            badge_icon = "&#9888;"
            badge_text = f"LOW CONFIDENCE &mdash; {top1_name.upper()}"
        elif label_type == "viable":
            badge_cls  = "badge-viable"
            badge_icon = "&#9679;"
            badge_text = top1_name.upper()
        elif label_type == "nonviable":
            badge_cls  = "badge-nonviable"
            badge_icon = "&#9679;"
            badge_text = top1_name.upper()
        else:
            badge_cls  = "badge-unknown"
            badge_icon = "?"
            badge_text = top1_name.upper()

        st.markdown(
            f'<div class="result-badge {badge_cls}">'
            f'<span>{badge_icon}</span> {badge_text}'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Confidence bar — FIX: use helper function, no raw HTML injection
        effective_type = label_type if not low_conf else "unknown"
        st.markdown(confidence_bar_html(bar_pct, effective_type), unsafe_allow_html=True)

        # Stat chips
        gauge_color = (
            "#2d7a4f" if label_type == "viable" and not low_conf else
            "#c0392b" if label_type == "nonviable" and not low_conf else
            "#e67e22"
        )

        st.markdown(
            '<div class="stat-row">'
            f'<div class="stat-chip">'
            f'  <div class="stat-val" style="color:{gauge_color};">{bar_pct:.1f}%</div>'
            f'  <div class="stat-lbl">Top Conf.</div>'
            f'</div>'
            f'<div class="stat-chip">'
            f'  <div class="stat-val">{top1_idx}</div>'
            f'  <div class="stat-lbl">Class ID</div>'
            f'</div>'
            f'<div class="stat-chip">'
            f'  <div class="stat-val">{len(probs_dict)}</div>'
            f'  <div class="stat-lbl">Total Kelas</div>'
            f'</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        # Gauge
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

    # ── Probability Breakdown
    st.markdown("""
    <div style="margin-top:1.4rem;">
        <div class="sec-label">Probability Breakdown</div>
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


# ─────────────────────────────────────────────
#  BATCH SUMMARY
# ─────────────────────────────────────────────
if len(batch_records) > 1:
    st.markdown("<hr class='tz-divider'>", unsafe_allow_html=True)
    st.markdown("""
    <div class="sec-label">Batch Summary</div>
    <div class="sec-title">Ringkasan Hasil Batch</div>
    """, unsafe_allow_html=True)

    df_batch = pd.DataFrame(batch_records)
    total    = len(df_batch)
    viable   = (df_batch["Tipe"].str.lower() == "viable").sum()
    nonvia   = (df_batch["Tipe"].str.lower() == "nonviable").sum()
    unknown  = total - viable - nonvia
    avg_conf = df_batch["Confidence (%)"].mean()

    chips_data = [
        (str(total),         "Total Gambar",  "#1a2e1e"),
        (str(viable),        "Viable",         "#2d7a4f"),
        (str(nonvia),        "Non-Viable",     "#c0392b"),
        (str(unknown),       "Unknown",        "#e67e22"),
        (f"{avg_conf:.1f}%", "Avg Confidence", "#1a2e1e"),
    ]
    c1, c2, c3, c4, c5 = st.columns(5)
    for col, (val, lbl, clr) in zip([c1, c2, c3, c4, c5], chips_data):
        with col:
            st.markdown(
                f'<div class="stat-chip">'
                f'  <div class="stat-val" style="color:{clr};">{val}</div>'
                f'  <div class="stat-lbl">{lbl}</div>'
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
            marker_colors=["#2d7a4f", "#c0392b", "#e67e22"],
            hole=0.58,
            textinfo="label+percent",
            textfont={"color": "#1a2e1e", "size": 12, "family": "Plus Jakarta Sans"},
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
                font_color="#1a2e1e",
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

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<hr class='tz-divider'>
<div style="text-align:center;font-size:0.7rem;color:#6b8a70;
            padding:0.5rem 0 1rem 0;font-family:'IBM Plex Mono',monospace;">
    Tetrazolium Viability Test &nbsp;&middot;&nbsp; Seed Analysis Suite &nbsp;&middot;&nbsp; &copy; 2025
</div>
""", unsafe_allow_html=True)
