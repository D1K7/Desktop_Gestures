import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import tensorflow as tf

# 1. Load your trained model and labels
model = tf.keras.models.load_model('gesture_recognizer.keras')
class_names = np.load('classes.npy', allow_pickle=True)

# 2. Setup MediaPipe Tasks
model_path = 'hand_landmarker.task'
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    running_mode=vision.RunningMode.VIDEO # Optimized for live streams
)
detector = vision.HandLandmarker.create_from_options(options)

def normalize_live_landmarks(landmarks):
    """Same normalization logic used during training"""
    coords = np.array([[lm.x, lm.y] for lm in landmarks])
    # Translation: Center by wrist
    coords = coords - coords[0]
    # Scaling: Unit size
    max_val = np.max(np.abs(coords))
    if max_val > 0:
        coords = coords / max_val
    return coords.flatten().reshape(1, -1) # Shape (1, 42) for ANN input

# 3. Open Webcam
cap = cv2.VideoCapture(0)
timestamp = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    
    # Process frame
    timestamp += 1
    result = detector.detect_for_video(mp_image, timestamp)

    if result.hand_landmarks:
        # Get normalized coordinates
        input_data = normalize_live_landmarks(result.hand_landmarks[0])
        
        # Predict gesture
        prediction = model.predict(input_data, verbose=0)
        class_id = np.argmax(prediction)
        confidence = prediction[0][class_id]
        
        gesture_name = class_names[class_id]

        # Display result if confidence is high enough
        if confidence > 0.8:
            cv2.putText(frame, f"{gesture_name} ({int(confidence*100)}%)", 
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Gesture Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()