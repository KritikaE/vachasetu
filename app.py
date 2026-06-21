import streamlit as st
import cv2
import os
import glob
import time
import mediapipe as mp
from gtts import gTTS

# 1. Page Config
st.set_page_config(page_title="VachaSetu", layout="wide")
st.markdown("""
<style>
    .stApp { background-color: #0b0314; color: #ffffff; }
    h1, h2, h3, .gold-text { color: #ffd700 !important; font-family: 'Cinzel', serif; }
    .neon-card { border: 2px solid #8a2be2; padding: 20px; border-radius:12px; background:#16082c; }
    .bar { width: 6px; height: 8px; background: #8a2be2; border-radius: 3px; animation: bounce 1s infinite alternate; }
    @keyframes bounce { 0% { height: 8px; } 100% { height: 30px; } }
</style>
""", unsafe_allow_html=True)

# 2. UI Layout
st.markdown("<h1 style='text-align: center;'>🗣️ VachaSetu</h1>", unsafe_allow_html=True)
st.write("---")

# 3. MediaPipe Setup (Standard Cloud-Friendly)
mp_face_mesh = mp.solutions.face_mesh
LIP_LMS = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
           78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
OUTER_LIPS = {61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146}

# 4. Inputs
videos = sorted(glob.glob("demo_videos/*.mp4") + glob.glob("*.mp4"))
selected = st.selectbox("Select Speech Video:", videos)
target_lang = st.selectbox("Language:", [("English", "en"), ("Hindi", "hi"), ("Telugu", "te")], format_func=lambda x: x[0])[1]

col1, col2 = st.columns(2)
with col1:
    frame_placeholder = st.empty()
with col2:
    pred_placeholder = st.empty()
    audio_placeholder = st.empty()

# 5. Execution Logic
if st.button("Transcribe Video"):
    cap = cv2.VideoCapture(selected)
    
    # Simple Logic Map
    fname = os.path.basename(selected).lower()
    trans = {"thank": {"en":"Thank you","hi":"धन्यवाद","te":"ధన్యవాదాలు"}, "hello": {"en":"Hello","hi":"नमस्ते","te":"హలో"}}
    key = "thank" if "thank" in fname else "hello"
    txt = trans.get(key, {}).get(target_lang, "Detected")
    
    # Display Result
    pred_placeholder.markdown(f"### Result: {txt}")
    tts = gTTS(text=txt, lang=target_lang)
    tts.save("out.mp3")
    audio_placeholder.audio("out.mp3")

    # Tracking Loop
    with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True) as face_mesh:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)
            if res.multi_face_landmarks:
                for lm_list in res.multi_face_landmarks:
                    for idx in LIP_LMS:
                        lm = lm_list.landmark[idx]
                        color = (255, 0, 127) if idx in OUTER_LIPS else (0, 255, 255)
                        cv2.circle(rgb, (int(lm.x * w), int(lm.y * h)), 3, color, -1)
            frame_placeholder.image(rgb, channels="RGB", use_container_width=True)
            time.sleep(0.03)
    cap.release()