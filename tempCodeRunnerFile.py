import cv2
import mediapipe as mp
import numpy as np
from pyfirmata import Arduino, util
import time

# Arduino Setup
board = Arduino('COM8')  # Change COM3 to your Arduino's port (e.g., '/dev/ttyUSB0' on Linux)
servo_pin = board.get_pin('d:9:s')  # digital pin 9 as servo

time.sleep(2)  # Wait for Arduino to initialize

# Video Capture
cap = cv2.VideoCapture(1)  # Use 0 for default cam
screen_w, screen_h = 640, 480  # Set virtual screen size to map servo range

# MediaPipe Setup
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Eye landmark indexes
RIGHT_EYE_INDEX = [33, 133]     # Approx corners of right eye
RIGHT_EYE_TOP_BOTTOM = [159, 145]  # For blink detection

# Utility function
def euclidean_distance(pt1, pt2):
    return np.linalg.norm(np.array(pt1) - np.array(pt2))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        h, w, _ = frame.shape
        landmarks = results.multi_face_landmarks[0].landmark

        # Draw face mesh
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1)
            )

        # Get eye center (right eye)
        right_eye = [landmarks[i] for i in RIGHT_EYE_INDEX]
        right_eye_center = np.mean([(int(p.x * w), int(p.y * h)) for p in right_eye], axis=0)

        # Blink detection
        top = np.array([landmarks[RIGHT_EYE_TOP_BOTTOM[0]].x * w, landmarks[RIGHT_EYE_TOP_BOTTOM[0]].y * h])
        bottom = np.array([landmarks[RIGHT_EYE_TOP_BOTTOM[1]].x * w, landmarks[RIGHT_EYE_TOP_BOTTOM[1]].y * h])
        blink_dist = euclidean_distance(top, bottom)

        # Convert eye x-coordinate to servo angle (0 to 180)
        servo_angle = np.interp(right_eye_center[0], [200, 440], [180,0])
        servo_pin.write(servo_angle)

        # Optional: display angle on frame
        cv2.putText(frame, f"Servo Angle: {int(servo_angle)}", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # Blink action (optional)
        if blink_dist < 5:
            cv2.putText(frame, "Blink Detected", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Eye to Servo Tracker", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
