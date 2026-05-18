import streamlit as st
import cv2
from deepface import DeepFace
import numpy as np
import time
from PIL import Image
import io

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Emotion Detector",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght=400;700&family=DM+Sans:wght=300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d0d0f;
    color: #e0e0e0;
}

h1, h2, h3 { font-family: 'Space Mono', monospace; }

.stApp { background-color: #0d0d0f; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #13131a;
    border-right: 1px solid #2a2a3a;
}

/* Metric cards */
.metric-card {
    background: #1a1a24;
    border: 1px solid #2a2a3a;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
    font-family: 'Space Mono', monospace;
}
.metric-label {
    font-size: 0.72rem;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
.metric-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #e0e0e0;
}

/* Emotion bar */
.bar-wrap { margin-bottom: 8px; }
.bar-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #aaa;
    display: flex;
    justify-content: space-between;
    margin-bottom: 3px;
}
.bar-track {
    height: 10px;
    background: #222232;
    border-radius: 5px;
    overflow: hidden;
}
.bar-fill {
    height: 100%;
    border-radius: 5px;
    transition: width 0.3s ease;
}

/* Dominant emotion badge */
.emotion-badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 999px;
    font-family: 'Space Mono', monospace;
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    margin-bottom: 12px;
}

/* Screenshot button */
.stDownloadButton > button {
    background: #1a1a2e;
    border: 1px solid #3a3a5a;
    color: #c0c0e0;
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    transition: all 0.2s;
}
.stDownloadButton > button:hover {
    border-color: #7c7cff;
    color: #fff;
    background: #1f1f3a;
}

/* Main image */
[data-testid="stImage"] img {
    border-radius: 14px;
    border: 1px solid #2a2a3a;
}

/* Hide default Streamlit branding */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
EMOTION_COLORS_HEX = {
    "happy":    "#00dc78",
    "sad":      "#c85050",
    "angry":    "#e03c14",
    "surprise": "#00c8f0",
    "fear":     "#8c00c8",
    "disgust":  "#28a028",
    "neutral":  "#b4b4b4",
}
EMOTION_COLORS_BGR = {
    "happy":    (120, 220, 0),    
    "sad":      (80, 80, 200),    
    "angry":    (20, 60, 224),    
    "surprise": (240, 200, 0),    
    "fear":     (200, 0, 140),    
    "disgust":  (40, 160, 40),    
    "neutral":  (180, 180, 180),  
}
EMOJI_MAP = {
    "happy": "😄", "sad": "😢", "angry": "😠",
    "surprise": "😮", "fear": "😨", "disgust": "🤢", "neutral": "😐",
}

# ── CV helpers ────────────────────────────────────────────────────────────────
def draw_rounded_rect(img, pt1, pt2, color, thickness=2, r=12):
    x1, y1 = pt1; x2, y2 = pt2
    cv2.line(img, (x1+r, y1), (x2-r, y1), color, thickness)
    cv2.line(img, (x1+r, y2), (x2-r, y2), color, thickness)
    cv2.line(img, (x1, y1+r), (x1, y2-r), color, thickness)
    cv2.line(img, (x2, y1+r), (x2, y2-r), color, thickness)
    cv2.ellipse(img, (x1+r, y1+r), (r,r), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2-r, y1+r), (r,r), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1+r, y2-r), (r,r),  90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2-r, y2-r), (r,r),   0, 0, 90, color, thickness)

def annotate_frame(frame, results):
    display = frame.copy()
    for face in results:
        region   = face.get("region", {})
        fx, fy   = region.get("x", 0), region.get("y", 0)
        fw, fh   = region.get("w", 0), region.get("h", 0)
        dominant = face.get("dominant_emotion", "neutral")
        color    = EMOTION_COLORS_BGR.get(dominant, (180,180,180))
        emoji    = EMOJI_MAP.get(dominant, "")

        draw_rounded_rect(display, (fx, fy), (fx+fw, fy+fh), color, 2)
        label = f"{dominant.upper()} {emoji}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.65, 1)
        cv2.rectangle(display, (fx, fy-th-12), (fx+tw+10, fy), color, -1)
        cv2.putText(display, label, (fx+5, fy-5),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, (255,255,255), 1)
    return display

def frame_to_pil(frame):
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

def pil_to_bytes(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()

def analyze_frame(frame):
    try:
        results = DeepFace.analyze(
            frame, actions=["emotion"],
            enforce_detection=False, silent=True
        )
        return results if isinstance(results, list) else [results]
    except Exception:
        return []

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎭 Emotion Detector")
    st.markdown("---")

    source = st.radio("Input source", ["📷 Webcam", "🖼️ Upload image"], index=0)
    st.markdown("---")

    frame_skip = st.slider("Analyse every N frames", 1, 10, 3,
        help="Higher = faster but less responsive (webcam mode)")
    st.markdown("---")

    st.markdown(
        "<div style='font-size:0.75rem;color:#555;font-family:Space Mono,monospace'>"
        "Powered by DeepFace · OpenCV<br>Press screenshot to save results"
        "</div>",
        unsafe_allow_html=True
    )

# ── Emotion sidebar panel (rendered after analysis) ───────────────────────────
def render_emotion_panel(results):
    if not results:
        st.info("No face detected.")
        return
    for i, face in enumerate(results):
        dominant = face.get("dominant_emotion", "neutral")
        emotions  = face.get("emotion", {})
        hex_col   = EMOTION_COLORS_HEX.get(dominant, "#b4b4b4")
        emoji     = EMOJI_MAP.get(dominant, "")

        if len(results) > 1:
            st.markdown(f"**Face {i+1}**")

        st.markdown(
            f"<div class='emotion-badge' style='background:{hex_col}22;"
            f"border:1px solid {hex_col};color:{hex_col}'>"
            f"{emoji} {dominant.upper()}</div>",
            unsafe_allow_html=True
        )

        # Emotion bars
        sorted_emos = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
        bars_html = ""
        for emo, score in sorted_emos:
            col = EMOTION_COLORS_HEX.get(emo, "#aaa")
            bars_html += f"""
            <div class='bar-wrap'>
              <div class='bar-label'><span>{emo}</span><span>{score:.1f}%</span></div>
              <div class='bar-track'>
                <div class='bar-fill' style='width:{score:.1f}%;background:{col}'></div>
              </div>
            </div>"""
        st.markdown(bars_html, unsafe_allow_html=True)
        if i < len(results)-1:
            st.markdown("---")

# ── Main layout ───────────────────────────────────────────────────────────────
col_vid, col_info = st.columns([3, 1], gap="large")

with col_vid:
    st.markdown("### Live Feed")
    img_placeholder = st.empty()
    metrics_row = st.columns(3)
    fps_box      = metrics_row[0].empty()
    faces_box    = metrics_row[1].empty()
    dominant_box = metrics_row[2].empty()
    dl_placeholder = st.empty()

with col_info:
    st.markdown("### Analysis")
    panel_placeholder = st.empty()

# ── Image upload mode ─────────────────────────────────────────────────────────
if source == "🖼️ Upload image":
    uploaded = st.file_uploader("Upload a photo", type=["jpg","jpeg","png","webp"])
    if uploaded:
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        results = analyze_frame(frame)
        annotated = annotate_frame(frame, results)
        pil_ann = frame_to_pil(annotated)

        with col_vid:
            img_placeholder.image(pil_ann, use_container_width=True)
            faces = len(results)
            dom = results[0].get("dominant_emotion","—").upper() if results else "—"
            fps_box.markdown(
                f"<div class='metric-card'><div class='metric-label'>Mode</div>"
                f"<div class='metric-value'>Photo</div></div>", unsafe_allow_html=True)
            faces_box.markdown(
                f"<div class='metric-card'><div class='metric-label'>Faces</div>"
                f"<div class='metric-value'>{faces}</div></div>", unsafe_allow_html=True)
            dominant_box.markdown(
                f"<div class='metric-card'><div class='metric-label'>Dominant</div>"
                f"<div class='metric-value' style='font-size:1.1rem'>{dom}</div></div>",
                unsafe_allow_html=True)
            dl_placeholder.download_button(
                "📸 Download annotated image", pil_to_bytes(pil_ann),
                file_name="emotion_result.png", mime="image/png"
            )
        with col_info:
            with panel_placeholder.container():
                render_emotion_panel(results)
    else:
        with col_vid:
            img_placeholder.markdown(
                "<div style='height:340px;display:flex;align-items:center;"
                "justify-content:center;border:1px dashed #333;border-radius:14px;"
                "color:#444;font-family:Space Mono,monospace;font-size:0.85rem'>"
                "↑ Upload an image to begin</div>", unsafe_allow_html=True
            )

# ── Webcam mode ───────────────────────────────────────────────────────────────
else:
    run = st.checkbox("▶ Start camera", value=False)
    
    if "last_screenshot" not in st.session_state:
        st.session_state.last_screenshot = None

    if not run:
        with col_vid:
            img_placeholder.markdown(
                "<div style='height:340px;display:flex;align-items:center;"
                "justify-content:center;border:1px dashed #333;border-radius:14px;"
                "color:#444;font-family:Space Mono,monospace;font-size:0.85rem'>"
                "Enable camera to begin</div>", unsafe_allow_html=True
            )
        if st.session_state.last_screenshot is not None:
            dl_placeholder.download_button(
                "📸 Download last webcam screenshot", 
                st.session_state.last_screenshot,
                file_name="webcam_emotion.png", 
                mime="image/png"
            )
    else:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

        frame_count = 0
        last_results = []
        t_prev = time.time()
        stop_btn = st.button("⏹ Stop camera")

        try:
            while cap.isOpened() and not stop_btn:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Could not read from camera.")
                    break

                frame_count += 1
                t_now = time.time()
                fps = 1 / max(t_now - t_prev, 1e-6)
                t_prev = t_now

                if frame_count % frame_skip == 0:
                    last_results = analyze_frame(frame)

                annotated = annotate_frame(frame, last_results)
                pil_frame = frame_to_pil(annotated)

                st.session_state.last_screenshot = pil_to_bytes(pil_frame)

                faces = len(last_results)
                dom = last_results[0].get("dominant_emotion","—").upper() if last_results else "—"

                with col_vid:
                    img_placeholder.image(pil_frame, use_container_width=True)
                    fps_box.markdown(
                        f"<div class='metric-card'><div class='metric-label'>FPS</div>"
                        f"<div class='metric-value'>{fps:.1f}</div></div>", unsafe_allow_html=True)
                    faces_box.markdown(
                        f"<div class='metric-card'><div class='metric-label'>Faces</div>"
                        f"<div class='metric-value'>{faces}</div></div>", unsafe_allow_html=True)
                    dominant_box.markdown(
                        f"<div class='metric-card'><div class='metric-label'>Dominant</div>"
                        f"<div class='metric-value' style='font-size:1.1rem'>{dom}</div></div>",
                        unsafe_allow_html=True)

                with col_info:
                    with panel_placeholder.container():
                        render_emotion_panel(last_results)
                        
        finally:
            cap.release()
            st.rerun()
