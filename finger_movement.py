import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe's hand detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Capture video from webcam
cap = cv2.VideoCapture(0)

# Function to calculate the movement of a point
def calculate_movement(prev_point, curr_point):
    if prev_point and curr_point:
        movement = np.linalg.norm(np.array(curr_point) - np.array(prev_point))
        return movement
    return 0

# Variables to store previous finger positions
prev_finger_positions = [None] * 5  # To store positions of 5 fingers

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Flip and convert the frame to RGB
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and find hands
    result = hands.process(rgb_frame)
    finger_movements = []  # To store movement data of fingers

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw landmarks on the frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get positions of each fingertip (index 4, 8, 12, 16, 20 correspond to thumb, index, middle, ring, pinky)
            fingertip_indices = [4, 8, 12, 16, 20]
            current_finger_positions = []

            for i, idx in enumerate(fingertip_indices):
                x = hand_landmarks.landmark[idx].x * frame.shape[1]
                y = hand_landmarks.landmark[idx].y * frame.shape[0]
                current_finger_positions.append((x, y))
                
                # Calculate movement for each finger
                if prev_finger_positions[i] is not None:
                    movement = calculate_movement(prev_finger_positions[i], (x, y))
                    finger_movements.append(movement)

                    # Display movement on the frame
                    cv2.putText(frame, f'Movement Finger {i+1}: {movement:.2f}', (10, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # Update previous finger positions
            prev_finger_positions = current_finger_positions

    # Display the output
    cv2.imshow("Finger Movement Detector", frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
