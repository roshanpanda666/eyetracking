import cv2
import mediapipe as mp
import numpy as np
from pyfirmata import Arduino, util
import time

# Arduino Setup
board = Arduino('COM8')  # Update this to your correct port
servo_horizontal = board.get_pin('d:9:s')   # Horizontal servo on pin 9
servo_vertical = board.get_pin('d:12:s')    # Vertical servo on pin 12
servo_blink = board.get_pin('d:13:s') 
time.sleep(2)  # Allow Arduino to initialize

# Video Capture
cap = cv2.VideoCapture(1)  # Change to 0 if needed
screen_w, screen_h = 640, 480

# MediaPipe Setup

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Eye landmark indexes
RIGHT_EYE_INDEX = [33, 133]
RIGHT_EYE_TOP_BOTTOM = [159, 145]

# Utility
def euclidean_distance(pt1, pt2):
    return np.linalg.norm(np.array(pt1) - np.array(pt2))

# Calibration
print("Calibration: Look straight ahead and press 'c'...")

calibration_done = False
neutral_eye_x = None
neutral_eye_y = None

while not calibration_done:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        h, w, _ = frame.shape
        landmarks = results.multi_face_landmarks[0].landmark
        right_eye = [landmarks[i] for i in RIGHT_EYE_INDEX]
        right_eye_center = np.mean([(int(p.x * w), int(p.y * h)) for p in right_eye], axis=0)

        cv2.putText(frame, "Calibrating... Look Straight & Press 'c'", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow("Calibration", frame)

        key = cv2.waitKey(1)
        if key == ord('c'):
            neutral_eye_x = right_eye_center[0]
            neutral_eye_y = right_eye_center[1]
            print(f"Calibrated neutral eye center: X={neutral_eye_x}, Y={neutral_eye_y}")
            calibration_done = True
            cv2.destroyWindow("Calibration")

# Set left/right and up/down bounds
x_range = 120
y_range = 80
min_x = neutral_eye_x - x_range
max_x = neutral_eye_x + x_range
min_y = neutral_eye_y - y_range
max_y = neutral_eye_y + y_range

print(f"Eye X Range: {min_x} to {max_x}")
print(f"Eye Y Range: {min_y} to {max_y}")

# Main Loop
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

        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1)
            )

        # Eye center
        right_eye = [landmarks[i] for i in RIGHT_EYE_INDEX]
        right_eye_center = np.mean([(int(p.x * w), int(p.y * h)) for p in right_eye], axis=0)
        eye_x, eye_y = right_eye_center

        # Blink detection
        top = np.array([landmarks[RIGHT_EYE_TOP_BOTTOM[0]].x * w, landmarks[RIGHT_EYE_TOP_BOTTOM[0]].y * h])
        bottom = np.array([landmarks[RIGHT_EYE_TOP_BOTTOM[1]].x * w, landmarks[RIGHT_EYE_TOP_BOTTOM[1]].y * h])
        blink_dist = euclidean_distance(top, bottom)

        # Map X and Y to servo angles
        servo_x = np.interp(eye_x, [min_x, max_x], [180, 0])  # Horizontal
        servo_y = np.interp(eye_y, [min_y, max_y], [0, 180])  # Vertical (invert Y so looking up means up)

        # Clamp angles
        servo_x = max(0, min(180, servo_x))
        servo_y = max(0, min(180, servo_y))

        # Move servos
        servo_horizontal.write(servo_x)     
        servo_vertical.write(servo_y)

        # Show info
        cv2.putText(frame, f"H: {int(servo_x)}°  V: {int(servo_y)}°", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        if blink_dist < 5:
            cv2.putText(frame, "Blink Detected", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            servo_blink.write(90)
            print("triggered")

    cv2.imshow("Eye-Controlled Dual Servo", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
