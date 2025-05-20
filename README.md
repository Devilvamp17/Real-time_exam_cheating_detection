# Real-Time Exam Cheating Detection

This project implements a real-time cheating detection system using object detection and face pose estimation. It leverages YOLOv8 for detecting suspicious objects (like phones, books, laptops) and MediaPipe for monitoring head orientation to detect when a user looks away from the screen.

---

## Features

- **Real-time webcam monitoring**
- Detects multiple suspicious items:
  - Person count (alerts if no person or multiple persons detected)
  - Mobile phones
  - Books
  - Remote controls and mice
  - Laptops
  - Earphones
- Face pose estimation to detect if the examinee is looking away from the screen
- Warnings and alerts with real-time feedback on the video feed
- Screenshots and logs are saved on suspicious behavior
- Command-line interface (CLI) for easy control of monitoring

---

## Requirements

- Python 3.8+
- `ultralytics` (YOLOv8)
- `opencv-python`
- `mediapipe`
- `click`
- `numpy`

Install dependencies via:

```bash
pip install ultralytics opencv-python mediapipe click numpy
