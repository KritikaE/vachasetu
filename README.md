# Vācha-Setu (Silent Speech Translator) 🗣️

Vācha-Setu (meaning "Voice Bridge" in Sanskrit) is an end-to-end AI prototype designed to capture real-time video of silent lip movements, extract spatial-temporal facial landmarks, classify the spoken phrase using a deep learning model, and synthesize vocalized audio output. It supports multilingual TTS audio rendering in English, Hindi, and Telugu.

---

## 🚀 Key Features
1. **Geometric Face Normalization**: Robust processing pipeline that translates, rotates, and scales facial landmarks (using outer eye corners and nose tip anchors) to be invariant to distance, camera perspective, and head tilt.
2. **Spatio-Temporal Deep Learning**: PyTorch architecture combining a spatial fully connected mapping layer with a Bidirectional LSTM network to classify dynamic lip contours.
3. **Multilingual Edge TTS**: Low-latency, non-blocking translation and speech routing for English, Hindi, and Telugu. Pregenerates audio files to ensure instant playback without network delays.
4. **Interactive Tracking HUD**: High-fidelity dark UI displaying a graphical HUD overlay showing the roll and scale corrected facial landmark tracking.
5. **Multiple Input Modalities**:
   - **Local Live Webcam (Real-time OpenCV)**: Captures continuous frame streams directly from local hardware using OpenCV. This is optimized for low-latency **localhost** operation.
   - **Interactive Simulator (Demo)**: Instantly test tracking, model inference, and audio translation using generated synthetic sequences without camera permissions (ideal for cloud reviewers).
   - **Webcam Snapshot**: Capture live snapshots to build up a temporal speaking sequence.
   - **Video Upload**: Transcribe uploaded silent video clips directly.

---

## 🛠️ Codebase Structure
```
├── app.py                  # Main Streamlit web application & UI
├── train.py                # Model training script
├── src/
│   ├── data_pipeline.py    # MediaPipe coordinate normalization & sequencing
│   ├── model.py            # PyTorch Spatial-Temporal CNN-LSTM network
│   └── audio_synthesis.py  # Translation mapping & audio cached TTS layer
├── tests/
│   └── test_pipeline.py    # Unit tests for testing shapes and math
├── requirements.txt        # System package dependencies
└── README.md               # User documentation & setup guide
```

---

## 💻 Local Quickstart

### 1. Clone the repository and navigate into it:
```bash
git clone <your-repository-url>
cd VachaSetu
```

### 2. Set up a virtual environment (recommended):
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install dependencies:
```bash
pip install -r requirements.txt
```

### 4. Run automated tests to verify math & tensor shapes:
```bash
$env:PYTHONPATH="."; python tests/test_pipeline.py
```
*(On Linux/macOS, use: `PYTHONPATH=. python tests/test_pipeline.py`)*

### 5. Pre-train the model weights:
```bash
$env:PYTHONPATH="."; python train.py
```
*(On Linux/macOS, use: `PYTHONPATH=. python train.py`)*

### 6. Launch the Streamlit dashboard:
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## ☁️ Cloud Deployment Instructions (e.g. Render / Vercel)

Vācha-Setu is fully optimized to run on cloud container systems like **Render** or **Vercel**. Since standard webcam capture (`cv2.VideoCapture(0)`) fails on cloud backends, Vācha-Setu uses browser-side camera capture via Streamlit's interface and includes a fully functional **Interactive Simulator** mode to demonstrate validation offline.

### Deployment on Render (Web Service)
1. **Create a New Web Service**: Connect your GitHub repository to your Render account.
2. **Environment & Runtime Settings**:
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt && python train.py` (This will install packages and automatically pre-train the model weights on startup)
   - **Start Command**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
3. **Environment Variables**:
   No external API tokens are required. However, you can configure:
   - `PYTHONPATH`: `.`
4. **Instance Type**:
   A free tier CPU instance is fully sufficient. The normalization logic, PyTorch network, and cached audio routing are optimized for ultra-low CPU footprint and execute in milliseconds.
