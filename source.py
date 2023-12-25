import mediapipe as mp
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)

cap = cv2.VideoCapture(0)

# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Make Detections
        results = holistic.process(image)
        
        # Recolor image back to BGR for rendering
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Check if both left and right hand landmarks are detected
        if results.left_hand_landmarks and results.right_hand_landmarks:
            # Draw landmarks for each hand
            for hand_landmarks in [results.left_hand_landmarks, results.right_hand_landmarks]:
                # Draw landmarks for each hand
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                          mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                                          )

                # Highlight the wrist for each hand
                wrist_landmark = hand_landmarks.landmark[0]
                wrist_x, wrist_y = int(wrist_landmark.x * frame.shape[1]), int((wrist_landmark.y + 0.05) * frame.shape[0])

                # Draw filled circle at the adjusted wrist position for each hand
                cv2.circle(image, (wrist_x, wrist_y), 8, (255, 0, 0), -1)

        cv2.imshow('Raw Webcam Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
