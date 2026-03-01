import cv2
import mediapipe as mp

# Setup MediaPipe
mp_holistic = mp.solutions.holistic 
mp_drawing = mp.solutions.drawing_utils 

def open_camera():
    # Try common Windows backends and camera indices.
    candidates = [
        
        (1, cv2.CAP_DSHOW),
        (2, cv2.CAP_DSHOW),
        (1, cv2.CAP_ANY),
        (2, cv2.CAP_ANY),
        
        # 2. Leave the broken laptop camera for absolute last
        (0, cv2.CAP_DSHOW), 
        (0, cv2.CAP_ANY),
    ]
    for index, backend in candidates:
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            continue
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f"Using camera index={index}, backend={backend}")
            return cap
        cap.release()
    return None

cap = open_camera()

if cap is None:
    raise RuntimeError("Camera not accessible. Close other apps using camera and allow camera permission in Windows Settings.")

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        
        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Make Detections
        results = holistic.process(image)
        
        # Recolor back to BGR for viewing
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Draw face and hand landmarks
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        cv2.imshow('PlexMeet Feed', image)

        # Press 'q' to quit
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
