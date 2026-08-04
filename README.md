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

A Streamlit app that detects facial emotions (angry, disgust, fear, happy, neutral, sad, surprise)
in real time using your browser's webcam (via WebRTC). A CNN model classifies each detected face
and a live mood meter tracks emotion counts during the session, with an option to download the
session log as an Excel file.

Click **Start** on the camera widget and allow camera access when prompted. Video is processed
frame-by-frame directly in the app; nothing is stored server-side beyond the in-memory session
counts.
