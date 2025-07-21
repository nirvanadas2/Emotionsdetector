from flask import Flask, render_template, Response, jsonify
import cv2
from keras.models import model_from_json
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model
json_file = open("facialemotionmodel.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("facialemotionmodel.weights.h5")


# Load the Haar Cascade
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Define labels
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
emotion_counts = {label: 0 for label in labels.values()}

# Define output and frame count
output = []
frame_count = 0
save_interval = 100  # Adjust this value as needed

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

def process_frame(frame):
    global frame_count
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (p, q, r, s) in faces:
        image = gray[q:q+s, p:p+r]
        cv2.rectangle(frame, (p, q), (p+r, q+s), (97, 218, 251), 2)
        image = cv2.resize(image, (48, 48))
        img = extract_features(image)
        pred = model.predict(img)
        prediction_label = labels[pred.argmax()]
        output.append(prediction_label)
        emotion_counts[prediction_label] += 1
        cv2.putText(frame, '% s' % (prediction_label), (p-10, q-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 255, 255))

    frame_count += 1
    if frame_count >= save_interval:
        save_output_to_excel()
        frame_count = 0

    return frame

def generate_frames():
    webcam = cv2.VideoCapture(0)
    webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    if not webcam.isOpened():
        print("Could not open webcam")
        return

    while True:
        success, frame = webcam.read()
        if not success:
            break
        else:
            frame = process_frame(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def save_output_to_excel():
    df = pd.DataFrame(output, columns=["Emotion"])
    df.to_excel('output.xlsx', index=False)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_page')
def video_page():
    return render_template('video.html')

@app.route('/emotion_data')
def emotion_data():
    return jsonify(emotion_counts)

@app.route('/reset_emotion_data', methods=['POST'])
def reset_emotion_data():
    global emotion_counts
    emotion_counts = {label: 0 for label in labels.values()}
    return jsonify({"status": "success"})

if __name__ == "__main__":
    app.run(debug=True)
