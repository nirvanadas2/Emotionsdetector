from collections import deque
from io import BytesIO

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from keras.models import model_from_json

LABELS = {0: "angry", 1: "disgust", 2: "fear", 3: "happy", 4: "neutral", 5: "sad", 6: "surprise"}


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


def detect_emotions(img_bgr, model, face_cascade):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    detections = []
    for (p, q, r, s) in faces:
        face_img = cv2.resize(gray[q:q + s, p:p + r], (48, 48))
        pred = model(extract_features(face_img), training=False).numpy()
        label = LABELS[int(pred.argmax())]
        detections.append((p, q, r, s, label))

    annotated = img_bgr.copy()
    for (p, q, r, s, label) in detections:
        cv2.rectangle(annotated, (p, q), (p + r, q + s), (97, 218, 251), 2)
        cv2.putText(annotated, label, (p, max(q - 10, 20)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (255, 255, 255), 2)

    return annotated, detections


if "emotion_counts" not in st.session_state:
    st.session_state.emotion_counts = {label: 0 for label in LABELS.values()}
if "history" not in st.session_state:
    st.session_state.history = deque(maxlen=2000)
if "last_photo_id" not in st.session_state:
    st.session_state.last_photo_id = None

st.set_page_config(page_title="Emotion Detection", layout="wide")
st.title("Facial Emotion Detection")
st.caption("Take a photo below and a CNN model will detect the emotion on each face in it.")

photo_col, meter_col = st.columns([2, 1])

with photo_col:
    photo = st.camera_input("Take a photo")

    if photo is not None and photo.file_id != st.session_state.last_photo_id:
        st.session_state.last_photo_id = photo.file_id

        model = load_model()
        face_cascade = load_face_cascade()

        file_bytes = np.frombuffer(photo.getvalue(), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        annotated, detections = detect_emotions(img_bgr, model, face_cascade)

        for (_, _, _, _, label) in detections:
            st.session_state.emotion_counts[label] += 1
            st.session_state.history.append(label)

        st.session_state.last_annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        st.session_state.last_detection_count = len(detections)

    if "last_annotated" in st.session_state:
        st.image(st.session_state.last_annotated, channels="RGB", use_container_width=True)
        if st.session_state.last_detection_count == 0:
            st.warning("No face detected in that photo -- try moving closer or improving lighting.")

with meter_col:
    st.subheader("Mood Meter")

    counts = st.session_state.emotion_counts
    history = list(st.session_state.history)

    df = pd.DataFrame(sorted(counts.items(), key=lambda kv: -kv[1]), columns=["Emotion", "Count"])
    st.dataframe(df, hide_index=True, use_container_width=True)

    if history:
        buffer = BytesIO()
        pd.DataFrame(history, columns=["Emotion"]).to_excel(buffer, index=False)
        st.download_button(
            "Download session data (.xlsx)",
            data=buffer.getvalue(),
            file_name="emotion_session.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    else:
        st.info("Take a photo to start building the mood meter.")
