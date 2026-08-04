import base64

import cv2
import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, render_template, request
from keras.models import model_from_json

app = Flask(__name__)

# Load the model
with open("facialemotionmodel.json", "r") as json_file:
    model_json = json_file.read()
model = model_from_json(model_json)
model.load_weights("facialemotionmodel.weights.h5")

# Load the Haar Cascade
haar_file = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haar_file)

# Define labels
labels = {0: "angry", 1: "disgust", 2: "fear", 3: "happy", 4: "neutral", 5: "sad", 6: "surprise"}
emotion_counts = {label: 0 for label in labels.values()}

# Define output and frame count
output = []
frame_count = 0
save_interval = 100  # Adjust this value as needed


def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0


def decode_frame(data_url):
    header, encoded = data_url.split(",", 1)
    binary = base64.b64decode(encoded)
    arr = np.frombuffer(binary, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def process_frame(frame):
    global frame_count
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    detections = []
    for (p, q, r, s) in faces:
        image = gray[q:q + s, p:p + r]
        image = cv2.resize(image, (48, 48))
        img = extract_features(image)
        pred = model.predict(img, verbose=0)
        prediction_label = labels[pred.argmax()]
        output.append(prediction_label)
        emotion_counts[prediction_label] += 1
        detections.append({
            "box": [int(p), int(q), int(r), int(s)],
            "label": prediction_label,
        })

    frame_count += 1
    if frame_count >= save_interval:
        save_output_to_excel()
        frame_count = 0

    return detections


def save_output_to_excel():
    df = pd.DataFrame(output, columns=["Emotion"])
    df.to_excel("output.xlsx", index=False)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_page")
def video_page():
    return render_template("video.html")


@app.route("/predict_frame", methods=["POST"])
def predict_frame():
    data = request.get_json(silent=True) or {}
    image_data = data.get("image")
    if not image_data:
        return jsonify({"error": "no image provided"}), 400

    try:
        frame = decode_frame(image_data)
    except Exception:
        return jsonify({"error": "could not decode image"}), 400

    if frame is None:
        return jsonify({"error": "could not decode image"}), 400

    detections = process_frame(frame)
    return jsonify({"detections": detections})


@app.route("/emotion_data")
def emotion_data():
    return jsonify(emotion_counts)


@app.route("/reset_emotion_data", methods=["POST"])
def reset_emotion_data():
    global emotion_counts
    emotion_counts = {label: 0 for label in labels.values()}
    return jsonify({"status": "success"})


if __name__ == "__main__":
    app.run(debug=True)
