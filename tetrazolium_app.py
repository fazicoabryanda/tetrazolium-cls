import streamlit as st
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO
import os
import torch
import pandas as pd
from io import BytesIO
import base64
import plotly.graph_objects as go
import plotly.express as px
from werkzeug.utils import secure_filename

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="TZ Viability Test",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS  –  dark lab / scientific theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Root palette ── */
:root {
    --bg:        #0d1117;
    --surface:   #161b22;
    --surface2:  #1f2937;
    --border:    #30363d;
    --accent:    #e84545;
    --accent2:   #ff6b6b;
    --viable:    #3fb950;
    --nonviable: #e84545;
    --text:      #e6edf3;
    --muted:     #8b949e;
    --card-glow: rgba(232,69,69,0.08);
}

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem 3rem !important; max-width: 1400px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Hero header ── */
.hero {
    background: linear-gradient(135deg, #1a0a0a 0%, #0d1117 50%, #0a1628 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(232,69,69,0.18) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
}
.hero-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    color: var(--accent);
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    color: var(--text);
    line-height: 1.2;
    margin: 0 0 0.6rem 0;
}
.hero-title span { color: var(--accent); }
.hero-subtitle {
    font-size: 0.95rem;
    color: var(--muted);
    max-width: 560px;
    line-height: 1.6;
}

/* ── Cards ── */
.tz-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 2px 16px var(--card-glow);
    transition: border-color 0.2s;
}
.tz-card:hover { border-color: #444d56; }

/* ── Section label ── */
.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.18em;
    color: var(--accent);
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}
.section-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 1rem;
}

/* ── Result badges ── */
.result-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.6rem 1.2rem;
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 1rem;
    font-weight: 700;
    margin: 0.5rem 0;
}
.badge-viable {
    background: rgba(63,185,80,0.15);
    border: 1.5px solid var(--viable);
    color: var(--viable);
}
.badge-nonviable {
    background: rgba(232,69,69,0.15);
    border: 1.5px solid var(--nonviable);
    color: var(--nonviable);
}
.badge-unknown {
    background: rgba(139,148,158,0.15);
    border: 1.5px solid var(--muted);
    color: var(--muted);
}

/* ── Confidence bar ── */
.conf-bar-wrap {
    margin: 1rem 0;
}
.conf-bar-label {
    font-size: 0.78rem;
    color: var(--muted);
    margin-bottom: 0.3rem;
    font-family: 'Space Mono', monospace;
}
.conf-bar-bg {
    background: var(--surface2);
    border-radius: 99px;
    height: 10px;
    width: 100%;
    overflow: hidden;
}
.conf-bar-fill-viable {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, #2ea043, #3fb950);
    transition: width 0.6s cubic-bezier(.4,0,.2,1);
}
.conf-bar-fill-nonviable {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, #b91c1c, #e84545);
    transition: width 0.6s cubic-bezier(.4,0,.2,1);
}
.conf-bar-fill-unknown {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, #4b5563, #8b949e);
}

/* ── Stat chips ── */
.stat-row { display: flex; gap: 1rem; flex-wrap: wrap; margin: 1rem 0; }
.stat-chip {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.6rem 1rem;
    text-align: center;
    min-width: 110px;
    flex: 1;
}
.stat-chip-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--text);
}
.stat-chip-label {
    font-size: 0.7rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.2rem;
}

/* ── Info box ── */
.info-box {
    background: rgba(232,69,69,0.07);
    border-left: 3px solid var(--accent);
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    margin: 0.8rem 0;
    font-size: 0.85rem;
    color: var(--muted);
    line-height: 1.6;
}

/* ── Streamlit widget overrides ── */
.stFileUploader > div {
    background: var(--surface2) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: 10px !important;
}
.stButton > button {
    background: var(--accent) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.05em !important;
    padding: 0.6rem 1.5rem !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

.stSlider > div > div > div { background: var(--accent) !important; }

div[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    overflow: hidden;
}

/* ── Image container ── */
.img-container {
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
    background: var(--surface2);
}

/* ── Divider ── */
.tz-divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 1.5rem 0;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
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
def pil_to_b64(img: Image.Image, fmt="PNG") -> str:
    buf = BytesIO()
    img.convert("RGB").save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def classify_label(name: str) -> str:
    n = name.lower()
    if "nonviable" in n or "non_viable" in n or "non-viable" in n:
        return "nonviable"
    if "viable" in n:
        return "viable"
    return "unknown"


def build_gauge(value: float, label: str, color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(value * 100, 2),
        number={"suffix": "%", "font": {"size": 28, "color": "#e6edf3",
                                         "family": "Space Mono"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#8b949e",
                     "tickfont": {"color": "#8b949e", "size": 10}},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "#1f2937",
            "bordercolor": "#30363d",
            "steps": [
                {"range": [0, 50],  "color": "#161b22"},
                {"range": [50, 75], "color": "#1a2030"},
                {"range": [75, 100],"color": "#1a2a1a" if color == "#3fb950" else "#2a1a1a"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.85,
                "value": value * 100,
            },
        },
        title={"text": label, "font": {"color": "#8b949e", "size": 12,
                                        "family": "DM Sans"}},
    ))
    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e6edf3",
    )
    return fig


def build_prob_bar(probs: dict, class_names: dict) -> go.Figure:
    labels, values, colors = [], [], []
    color_map = {"viable": "#3fb950", "nonviable": "#e84545", "unknown": "#8b949e"}
    for idx, prob in sorted(probs.items(), key=lambda x: -x[1]):
        name = class_names.get(idx, f"Class {idx}")
        labels.append(name)
        values.append(round(prob * 100, 3))
        colors.append(color_map.get(classify_label(name), "#8b949e"))

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=colors,
        text=[f"{v:.2f}%" for v in values],
        textposition="outside",
        textfont={"color": "#e6edf3", "size": 11, "family": "Space Mono"},
        hovertemplate="%{y}: %{x:.3f}%<extra></extra>",
    ))
    fig.update_layout(
        height=max(180, len(labels) * 55),
        margin=dict(l=10, r=80, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[0, 115], showgrid=True, gridcolor="#30363d",
                   tickfont={"color": "#8b949e"}, zeroline=False),
        yaxis=dict(tickfont={"color": "#e6edf3", "size": 12}),
        bargap=0.35,
    )
    return fig


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:1rem 0 0.5rem 0;">
        <div style="font-family:'Space Mono',monospace;font-size:0.65rem;
                    letter-spacing:0.18em;color:#e84545;text-transform:uppercase;
                    margin-bottom:0.3rem;">Tetrazolium Test</div>
        <div style="font-size:1.1rem;font-weight:600;">Configuration</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#30363d;margin:0.5rem 0 1rem 0;'>",
                unsafe_allow_html=True)

    st.markdown("**Model**")
    model_path_input = st.text_input(
        "Model path (.pt)", value=MODEL_PATH,
        help="Path ke file model klasifikasi Tetrazolium YOLOv11",
        label_visibility="collapsed"
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Confidence Threshold**")
    conf_thresh = st.slider(
        "conf", min_value=0.01, max_value=0.99,
        value=0.25, step=0.01,
        label_visibility="collapsed",
        help="Hasil di bawah threshold ini akan ditandai sebagai 'low confidence'."
    )
    st.caption(f"Threshold aktif: `{conf_thresh:.2f}`")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Batch Mode**")
    batch_mode = st.toggle("Aktifkan multi-gambar", value=False,
                           help="Upload beberapa gambar sekaligus untuk diklasifikasi batch.")

    st.markdown("<hr style='border-color:#30363d;margin:1.5rem 0 1rem 0;'>",
                unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:0.75rem;color:#8b949e;line-height:1.7;">
    <b style="color:#e6edf3;">Tetrazolium (TZ) Test</b><br>
    Uji viabilitas benih menggunakan garam tetrazolium (2,3,5-triphenyltetrazolium chloride).
    Benih viable akan berwarna merah karena aktivitas enzim dehidrogenase.
    <br><br>
    <b style="color:#3fb950;">● Viable</b> — jaringan aktif, berwarna merah<br>
    <b style="color:#e84545;">● Non-Viable</b> — jaringan mati, tidak berwarna
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
    <div class="hero-label">🧪 Seed Science & Technology</div>
    <h1 class="hero-title">Tetrazolium<span> Viability</span> Test</h1>
    <p class="hero-subtitle">
        Klasifikasi viabilitas benih secara otomatis menggunakan deep learning.
        Upload gambar hasil pewarnaan TZ — model akan menentukan apakah benih
        <strong>viable</strong> atau <strong>non-viable</strong> beserta skor kepercayaannya.
    </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  MODEL STATUS
# ─────────────────────────────────────────────
if model_err:
    st.markdown(f"""
    <div class="tz-card" style="border-color:#e84545;">
        <div class="section-label">⚠ Model Error</div>
        <code style="color:#e84545;font-size:0.85rem;">{model_err}</code>
        <div class="info-box" style="margin-top:0.8rem;">
            Pastikan file <code>tetrazolium_model.pt</code> berada di direktori yang sama
            dengan script ini, atau ubah path model di sidebar.
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    class_names = getattr(model, "names", {})
    class_list_html = " &nbsp;·&nbsp; ".join(
        [f'<span style="color:#e6edf3;">{v}</span>' for v in class_names.values()]
    ) if class_names else "—"
    st.markdown(f"""
    <div class="tz-card" style="border-color:#3fb950;padding:1rem 1.5rem;">
        <div style="display:flex;align-items:center;gap:0.6rem;">
            <span style="color:#3fb950;font-size:1.1rem;">✓</span>
            <span style="font-size:0.9rem;color:#3fb950;font-family:'Space Mono',monospace;">
                Model loaded</span>
            <span style="color:#30363d;">|</span>
            <span style="font-size:0.8rem;color:#8b949e;">Kelas: {class_list_html}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr class='tz-divider'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  UPLOADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="section-label">Input</div>
<div class="section-title">Upload Gambar TZ</div>
""", unsafe_allow_html=True)

accept_multiple = batch_mode
uploaded = st.file_uploader(
    "Drag & drop gambar hasil uji Tetrazolium di sini",
    type=["png", "jpg", "jpeg", "bmp", "tiff"],
    accept_multiple_files=accept_multiple,
    key="tz_uploader",
    label_visibility="collapsed",
)

if not batch_mode and uploaded is not None:
    uploaded = [uploaded]   # normalise ke list

if not uploaded:
    st.markdown("""
    <div class="info-box">
        📌 Upload gambar benih setelah proses pewarnaan tetrazolium. Format yang didukung:
        PNG, JPG, JPEG, BMP, TIFF.
        Aktifkan <strong>Batch Mode</strong> di sidebar untuk mengklasifikasi banyak gambar sekaligus.
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
    f"🔬  Klasifikasi {'Semua Gambar' if len(uploaded) > 1 else 'Gambar'}",
    use_container_width=True,
    type="primary",
)

if not run_btn:
    # Preview uploaded images
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
<div class="section-label">Results</div>
<div class="section-title">Hasil Klasifikasi</div>
""", unsafe_allow_html=True)

batch_records = []   # for summary table

for file_idx, uf in enumerate(uploaded):
    fname = secure_filename(uf.name)
    tmp_path = os.path.join(UPLOAD_DIR, f"tz_{file_idx}_{fname}")
    with open(tmp_path, "wb") as fh:
        fh.write(uf.getbuffer())

    if len(uploaded) > 1:
        st.markdown(f"""
        <div style="font-family:'Space Mono',monospace;font-size:0.75rem;
                    color:#8b949e;margin:1.2rem 0 0.4rem 0;">
            [{file_idx+1}/{len(uploaded)}] {fname}
        </div>
        """, unsafe_allow_html=True)

    col_img, col_res = st.columns([1, 1], gap="large")

    # ── Left: image preview
    with col_img:
        st.markdown('<div class="section-label">Input Image</div>', unsafe_allow_html=True)
        try:
            pil_img = Image.open(tmp_path)
            w, h = pil_img.size
            st.image(pil_img, use_container_width=True)
            st.markdown(f"""
            <div style="font-size:0.72rem;color:#8b949e;margin-top:0.4rem;
                        font-family:'Space Mono',monospace;">
                {fname} &nbsp;|&nbsp; {w}×{h} px
            </div>
            """, unsafe_allow_html=True)
        except UnidentifiedImageError:
            st.error("File gambar tidak valid atau rusak.")
            continue
        except Exception as e:
            st.error(f"Gagal membuka gambar: {e}")
            continue

    # ── Right: inference + results
    with col_res:
        st.markdown('<div class="section-label">Classification Result</div>',
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

        # Probability dict
        try:
            raw = probs_obj.data
            probs_list = raw.cpu().numpy().tolist() if hasattr(raw, "cpu") else list(raw)
            probs_dict = {i: p for i, p in enumerate(probs_list)}
        except Exception:
            probs_dict = {top1_idx: top1_conf}

        top1_name  = class_names.get(top1_idx, f"Class {top1_idx}")
        label_type = classify_label(top1_name)
        low_conf   = top1_conf < conf_thresh

        # Badge
        if low_conf:
            badge_cls, badge_icon = "badge-unknown", "⚠"
            badge_text = f"LOW CONFIDENCE — {top1_name}"
        elif label_type == "viable":
            badge_cls, badge_icon = "badge-viable", "●"
            badge_text = top1_name.upper()
        elif label_type == "nonviable":
            badge_cls, badge_icon = "badge-nonviable", "●"
            badge_text = top1_name.upper()
        else:
            badge_cls, badge_icon = "badge-unknown", "?"
            badge_text = top1_name.upper()

        st.markdown(f"""
        <div class="result-badge {badge_cls}">
            <span>{badge_icon}</span> {badge_text}
        </div>
        """, unsafe_allow_html=True)

        # Confidence bar
        bar_pct = top1_conf * 100
        bar_cls = (
            "conf-bar-fill-viable"    if label_type == "viable" and not low_conf else
            "conf-bar-fill-nonviable" if label_type == "nonviable" and not low_conf else
            "conf-bar-fill-unknown"
        )
        bar_color_inline = (
            "#3fb950" if label_type == "viable" and not low_conf else
            "#e84545" if label_type == "nonviable" and not low_conf else
            "#8b949e"
        )
        st.markdown(f"""
        <div class="conf-bar-wrap">
            <div class="conf-bar-label">
                Confidence Score &nbsp;
                <span style="color:{bar_color_inline};font-weight:700;">
                    {bar_pct:.2f}%
                </span>
                {'&nbsp; <span style="color:#8b949e;font-size:0.7rem;">'
                 f'(threshold: {conf_thresh*100:.0f}%)</span>' if low_conf else ''}
            </div>
            <div class="conf-bar-bg">
                <div class="{bar_cls}" style="width:{bar_pct:.1f}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Stat chips
        st.markdown(f"""
        <div class="stat-row">
            <div class="stat-chip">
                <div class="stat-chip-value" style="color:{bar_color_inline};">
                    {bar_pct:.1f}%</div>
                <div class="stat-chip-label">Top Conf.</div>
            </div>
            <div class="stat-chip">
                <div class="stat-chip-value">{top1_idx}</div>
                <div class="stat-chip-label">Class ID</div>
            </div>
            <div class="stat-chip">
                <div class="stat-chip-value">{len(probs_dict)}</div>
                <div class="stat-chip-label">Total Kelas</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Gauge chart
        gauge_color = (
            "#3fb950" if label_type == "viable" and not low_conf else
            "#e84545" if label_type == "nonviable" and not low_conf else
            "#8b949e"
        )
        st.plotly_chart(
            build_gauge(top1_conf, "Confidence", gauge_color),
            use_container_width=True, config={"displayModeBar": False}
        )

        # Record for batch summary
        batch_records.append({
            "File": fname,
            "Prediksi": top1_name,
            "Tipe": label_type.capitalize(),
            "Confidence (%)": round(bar_pct, 3),
            "Low Conf.": "Ya" if low_conf else "Tidak",
        })

    # ── Probability breakdown (full width)
    st.markdown("""
    <div style="margin-top:1.2rem;">
        <div class="section-label">Probability Breakdown</div>
    </div>
    """, unsafe_allow_html=True)

    prob_col1, prob_col2 = st.columns([3, 2], gap="large")

    with prob_col1:
        st.plotly_chart(
            build_prob_bar(probs_dict, class_names),
            use_container_width=True, config={"displayModeBar": False}
        )

    with prob_col2:
        # Table
        rows = []
        for idx in sorted(probs_dict, key=lambda x: -probs_dict[x]):
            nm = class_names.get(idx, f"Class {idx}")
            rows.append({
                "Kelas": nm,
                "Probabilitas": f"{probs_dict[idx]*100:.3f}%",
                "Status": (
                    "✅ Terpilih" if idx == top1_idx else
                    "⚠ Low Conf" if idx == top1_idx and low_conf else "—"
                )
            })
        df_probs = pd.DataFrame(rows)
        st.dataframe(df_probs, use_container_width=True, hide_index=True,
                     height=min(300, len(rows) * 50 + 50))

    if file_idx < len(uploaded) - 1:
        st.markdown("<hr class='tz-divider'>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  BATCH SUMMARY  (only when >1 image)
# ─────────────────────────────────────────────
if len(batch_records) > 1:
    st.markdown("<hr class='tz-divider'>", unsafe_allow_html=True)
    st.markdown("""
    <div class="section-label">Batch Summary</div>
    <div class="section-title">Ringkasan Hasil Batch</div>
    """, unsafe_allow_html=True)

    df_batch = pd.DataFrame(batch_records)

    total   = len(df_batch)
    viable  = (df_batch["Tipe"].str.lower() == "viable").sum()
    nonvia  = (df_batch["Tipe"].str.lower() == "nonviable").sum()
    unknown = total - viable - nonvia
    avg_conf = df_batch["Confidence (%)"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    chips = [
        (str(total),            "Total Gambar",  "#e6edf3"),
        (str(viable),           "Viable",         "#3fb950"),
        (str(nonvia),           "Non-Viable",     "#e84545"),
        (str(unknown),          "Unknown",        "#8b949e"),
        (f"{avg_conf:.1f}%",   "Avg Confidence", "#e6edf3"),
    ]
    for col, (val, lbl, clr) in zip([c1,c2,c3,c4,c5], chips):
        with col:
            st.markdown(f"""
            <div class="stat-chip">
                <div class="stat-chip-value" style="color:{clr};">{val}</div>
                <div class="stat-chip-label">{lbl}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_tbl, col_pie = st.columns([3, 2], gap="large")

    with col_tbl:
        st.dataframe(df_batch, use_container_width=True, hide_index=True)

    with col_pie:
        pie_labels = ["Viable", "Non-Viable", "Unknown"]
        pie_values = [viable, nonvia, unknown]
        pie_colors = ["#3fb950", "#e84545", "#8b949e"]
        fig_pie = go.Figure(go.Pie(
            labels=pie_labels, values=pie_values,
            marker_colors=pie_colors,
            hole=0.55,
            textinfo="label+percent",
            textfont={"color": "#e6edf3", "size": 12, "family": "DM Sans"},
            hovertemplate="%{label}: %{value} gambar (%{percent})<extra></extra>",
        ))
        fig_pie.update_layout(
            height=280,
            margin=dict(l=10, r=10, t=30, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            annotations=[dict(text=f"<b>{total}</b><br><span style='font-size:10px'>total</span>",
                              x=0.5, y=0.5, font_size=18,
                              font_color="#e6edf3", showarrow=False)],
        )
        st.plotly_chart(fig_pie, use_container_width=True,
                        config={"displayModeBar": False})

    # ── CSV Download
    st.markdown("<br>", unsafe_allow_html=True)
    csv_bytes = df_batch.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇  Download Hasil Batch (.csv)",
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
<div style="text-align:center;font-size:0.72rem;color:#8b949e;padding:0.5rem 0 1rem 0;
            font-family:'Space Mono',monospace;">
    Tetrazolium Viability Test &nbsp;·&nbsp; Seed Analysis Suite &nbsp;·&nbsp; ©2025
</div>
""", unsafe_allow_html=True)
