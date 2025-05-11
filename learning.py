import cv2
import mediapipe as mp
import numpy as np
from pyfirmata import Arduino, SERVO
import time

# Arduino Setup
board = Arduino('COM8')  # Replace COM3 with your port (e.g., '/dev/ttyUSB0' for Linux)
servo_pin = 9
board.digital[servo_pin].mode = SERVO
time.sleep(2)  # Wait for Arduino to initialize

def rotate_servo(angle):
    angle = int(np.clip(angle, 0, 180))
    board.digital[servo_pin].write(angle)
    print(f"Rotating to: {angle}°")

# Mediapipe Setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
LEFT_IRIS_INDEXES = [468, 469, 470, 471]

# Webcam
cap = cv2.VideoCapture(1)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        h, w, _ = frame.shape
        for face_landmarks in result.multi_face_landmarks:
            iris_coords = []
            for idx in LEFT_IRIS_INDEXES:
                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)
                iris_coords.append((x, y))
                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

            if iris_coords:
                cx = int(np.mean([pt[0] for pt in iris_coords]))
                cy = int(np.mean([pt[1] for pt in iris_coords]))

                cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

                # Map eye X position to servo angle (0 to 180)
                angle = np.interp(cx, [0, w], [0, 180])
                rotate_servo(angle)

    cv2.imshow("Iris Servo Control", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
