import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Init
cap = cv2.VideoCapture(1)  # Use 0 if your webcam is on default cam
screen_w, screen_h = pyautogui.size()

# MediaPipe setup
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

        # ✅ Draw full face mesh
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

        # Move mouse cursor
        screen_x = np.interp(right_eye_center[0], [0, w], [0, screen_w * 1.5])
        screen_y = np.interp(right_eye_center[1], [0, h], [0, screen_h * 1.5])

        pyautogui.moveTo(screen_x, screen_y)

        # Click on blink
        if blink_dist < 5:  # Adjust this threshold if needed
            pyautogui.click()
            cv2.putText(frame, "Blink Detected - Clicked", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Eye Tracker with Mesh", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release() 
cv2.destroyAllWindows()
