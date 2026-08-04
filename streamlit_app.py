import threading
import time
from collections import deque
from io import BytesIO

import av
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from keras.models import model_from_json
from streamlit_autorefresh import st_autorefresh
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer

LABELS = {0: "angry", 1: "disgust", 2: "fear", 3: "happy", 4: "neutral", 5: "sad", 6: "surprise"}

RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:openrelay.metered.ca:80"]},
            {
                "urls": ["turn:openrelay.metered.ca:80"],
                "username": "openrelayproject",
                "credential": "openrelayproject",
            },
            {
                "urls": ["turn:openrelay.metered.ca:443"],
                "username": "openrelayproject",
                "credential": "openrelayproject",
            },
            {
                "urls": ["turn:openrelay.metered.ca:443?transport=tcp"],
                "username": "openrelayproject",
                "credential": "openrelayproject",
            },
        ]
    }
)


@st.cache_resource
def load_model():
    with open("facialemotionmodel.json", "r") as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    model.load_weights("facialemotionmodel.weights.h5")
    return model


@st.cache_resource
def load_face_cascade():
    haar_file = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(haar_file)


def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0


class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = load_model()
        self.face_cascade = load_face_cascade()
        self.lock = threading.Lock()
        self.emotion_counts = {label: 0 for label in LABELS.values()}
        self.history = deque(maxlen=2000)
        self.last_detections = []

        self._frame_lock = threading.Lock()
        self._latest_frame = None
        self._stop = False
        self._worker = threading.Thread(target=self._infer_loop, daemon=True)
        self._worker.start()

    def _infer_loop(self):
        # Runs continuously in the background, always working on the newest
        # available frame. This is decoupled from recv() on purpose: on a
        # slow CPU, detection+inference can take longer than one video
        # frame's worth of time, and blocking recv() on it causes the
        # video to buffer/stutter. Here it just lags the overlay instead.
        while not self._stop:
            with self._frame_lock:
                frame = self._latest_frame
                self._latest_frame = None

            if frame is None:
                time.sleep(0.03)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            detections = []
            for (p, q, r, s) in faces:
                face_img = cv2.resize(gray[q:q + s, p:p + r], (48, 48))
                pred = self.model(extract_features(face_img), training=False).numpy()
                label = LABELS[int(pred.argmax())]

                with self.lock:
                    self.emotion_counts[label] += 1
                    self.history.append(label)

                detections.append((p, q, r, s, label))
            self.last_detections = detections

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        with self._frame_lock:
            self._latest_frame = img

        for (p, q, r, s, label) in self.last_detections:
            cv2.rectangle(img, (p, q), (p + r, q + s), (97, 218, 251), 2)
            cv2.putText(img, label, (p, max(q - 10, 20)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (255, 255, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def __del__(self):
        self._stop = True


st.set_page_config(page_title="Real-Time Emotion Detection", layout="wide")
st.title("Real-Time Facial Emotion Detection")
st.caption("Allow camera access below to start. Detected faces are boxed and labeled live.")

video_col, meter_col = st.columns([2, 1])

with video_col:
    ctx = webrtc_streamer(
        key="emotion-detection",
        video_processor_factory=EmotionProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={
            "video": {"width": {"ideal": 480}, "height": {"ideal": 360}},
            "audio": False,
        },
    )

with meter_col:
    st.subheader("Mood Meter")
    table_placeholder = st.empty()
    download_placeholder = st.empty()

    if ctx.state.playing:
        st_autorefresh(interval=1000, key="mood-meter-refresh")

    if ctx.video_processor:
        with ctx.video_processor.lock:
            counts = dict(ctx.video_processor.emotion_counts)
            history = list(ctx.video_processor.history)

        df = pd.DataFrame(sorted(counts.items(), key=lambda kv: -kv[1]), columns=["Emotion", "Count"])
        table_placeholder.dataframe(df, hide_index=True, use_container_width=True)

        if history:
            now = time.time()
            last_build = st.session_state.get("_excel_built_at", 0)
            if now - last_build > 5 or "_excel_bytes" not in st.session_state:
                buffer = BytesIO()
                pd.DataFrame(history, columns=["Emotion"]).to_excel(buffer, index=False)
                st.session_state["_excel_bytes"] = buffer.getvalue()
                st.session_state["_excel_built_at"] = now

            download_placeholder.download_button(
                "Download session data (.xlsx)",
                data=st.session_state["_excel_bytes"],
                file_name="emotion_session.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
    else:
        table_placeholder.info("Start the camera to see live emotion counts.")
