# Hand Gesture Classification using MediaPipe and Machine Learning

## 📌 Project Overview
This project focuses on classifying 18 different hand gestures using 3D hand landmarks extracted from the **HaGRID** dataset via **MediaPipe**. Instead of using raw images, the model is trained entirely on normalized and recentered hand landmark coordinates.

## 🎯 Objectives
- Classify 18 predefined hand gestures from hand keypoint data
- Train and evaluate multiple machine learning models
- Visualize hand keypoints and predictions
- Perform real-time prediction via webcam (frame-by-frame in Google Colab)
- Optionally test on uploaded video files

## 📂 Dataset
- **HaGRID (Hand Gesture Recognition Image Dataset)**
- 18 gesture classes (e.g., `peace`, `stop`, `call`, `thumbs_up`, ...)
- Each sample includes 21 landmarks per hand: `(x, y, z)` coordinates
- Data was exported to a `.csv` format after landmark extraction

## ⚙️ Preprocessing
- Recenter landmarks to make wrist `(0, 0)`
- Normalize `x` and `y` coordinates using the `x` distance from wrist to middle fingertip
- Keep `z` values as-is (already relative depth)
- Remove missing/null values, check for duplicates

## 📊 Visualization
- Visualize hand landmarks (dots) with sample labels
- Plotted individual samples to verify proper normalization

## 🤖 Model Training
Four supervised ML algorithms were used:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Random Forest Classifier
- Naive Bayes

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score

## 🏆 Best Model
The best-performing model was selected based on **F1-score on validation set**. This model was then used for all prediction tasks.

## 🎥 Real-Time Prediction in Google Colab
Due to Colab's limitations, traditional real-time webcam capture (`cv2.VideoCapture(0)`) is not possible.

Instead, the following workarounds were used:
- 📸 JavaScript-based webcam frame capture inside Colab
- 🌀 Simulated real-time gesture prediction with a looped capture button
- 🎬 Video file upload support with frame-by-frame gesture prediction

## 🧪 How to Run
1. Upload `hand_landmarks.csv`
2. Preprocess data and split into train/val/test
3. Train models and evaluate performance
4. Run the real-time loop using webcam **OR** upload a video file to test

## 📦 Requirements
- Python
- Google Colab (recommended)
- `mediapipe`, `opencv-python`, `scikit-learn`, `pandas`, `matplotlib`

## 📝 Notes
- Real-time prediction on Colab is done one frame at a time
- If running locally, you can enable full real-time video capture with OpenCV

## ✅ Project Status
Complete. Models trained, evaluated, and tested on webcam frames and video.

---

Feel free to fork this project, customize it for your own gestures, or integrate it with real-time applications! 🚀

