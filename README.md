---
title: Emotion Detection
emoji: 🙂
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Real-Time Facial Emotion Detection

A Flask app that detects facial emotions (angry, disgust, fear, happy, neutral, sad, surprise)
in real time using your browser's webcam. A CNN model classifies each detected face and a
live mood meter tracks emotion counts during the session.

Open the app, click **Start Video Feed**, and allow camera access when prompted. All video
processing happens per-frame on the server; your camera stream never leaves your browser except
as individual JPEG frames sent for prediction.
