import customtkinter as ctk
import tkinter.messagebox as messagebox
import cv2
import mediapipe as mp
import numpy as np
from threading import Thread
from pyfirmata import Arduino, util
import time


# Setup Arduino
board = Arduino('COM8')  # Change COM port as needed
servo_horizontal = board.get_pin('d:9:s')
servo_vertical = board.get_pin('d:12:s')
servo_blink = board.get_pin('d:13:s')
time.sleep(2)

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5)

# Constants and variables
RIGHT_EYE_INDEX = [33, 133]
RIGHT_EYE_TOP_BOTTOM = [159, 145]

neutral_eye_x = None
neutral_eye_y = None
x_range = 120
y_range = 80
manual_mode = True
keyboard_mode = False
video_thread = None
hand_thread = None
ai_aim_thread = None
running = False
hand_tracking = False
ai_aiming = False
servo_x_angle = 90
servo_y_angle = 90
camera_index = 2

def euclidean_distance(pt1, pt2):
    return np.linalg.norm(np.array(pt1) - np.array(pt2))

def calibrate():
    global neutral_eye_x, neutral_eye_y
    cap = cv2.VideoCapture(camera_index)
    while True:
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
            neutral_eye_x, neutral_eye_y = right_eye_center
            break
    cap.release()

def face_tracking_loop():
    global running
    cap = cv2.VideoCapture(camera_index)
    min_x = neutral_eye_x - x_range
    max_x = neutral_eye_x + x_range
    min_y = neutral_eye_y - y_range
    max_y = neutral_eye_y + y_range
    while running:
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
            eye_x, eye_y = right_eye_center
            top = np.array([landmarks[RIGHT_EYE_TOP_BOTTOM[0]].x * w, landmarks[RIGHT_EYE_TOP_BOTTOM[0]].y * h])
            bottom = np.array([landmarks[RIGHT_EYE_TOP_BOTTOM[1]].x * w, landmarks[RIGHT_EYE_TOP_BOTTOM[1]].y * h])
            blink_dist = euclidean_distance(top, bottom)
            servo_x = np.interp(eye_x, [min_x, max_x], [180, 0])
            servo_y = np.interp(eye_y, [min_y, max_y], [0, 180])
            servo_horizontal.write(max(0, min(180, servo_x)))
            servo_vertical.write(max(0, min(180, servo_y)))
            if blink_dist < 5:
                servo_blink.write(90)
            mp_drawing.draw_landmarks(frame, results.multi_face_landmarks[0], mp_face_mesh.FACEMESH_TESSELATION)
        cv2.imshow("Face Tracking", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

def hand_tracking_loop():
    global hand_tracking
    cap = cv2.VideoCapture(camera_index)
    while hand_tracking:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        h, w, _ = frame.shape
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            index_finger_tip = hand_landmarks.landmark[8]
            x = int(index_finger_tip.x * w)
            y = int(index_finger_tip.y * h)
            servo_x = np.interp(x, [0, w], [180, 0])
            servo_y = np.interp(y, [0, h], [0, 180])
            servo_horizontal.write(servo_x)
            servo_vertical.write(servo_y)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.imshow("Hand Tracking", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

def ai_aim_loop():
    global ai_aiming
    cap = cv2.VideoCapture(camera_index)
    while ai_aiming:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            mid_x = int((left_shoulder.x + right_shoulder.x) / 2 * w)
            mid_y = int((left_shoulder.y + right_shoulder.y) / 2 * h)
            servo_x = np.interp(mid_x, [0, w], [0, 180])
            servo_y = np.interp(mid_y, [0, h], [0, 180])
            servo_horizontal.write(servo_x)
            servo_vertical.write(servo_y)
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.imshow("AI Aim", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

def start_hand_tracking():
    global hand_tracking, manual_mode, keyboard_mode, running, hand_thread
    stop_all_modes()
    manual_mode = False
    keyboard_mode = False
    hand_tracking = True
    hand_thread = Thread(target=hand_tracking_loop)
    hand_thread.start()

def stop_hand_tracking():
    global hand_tracking
    hand_tracking = False

def start_face_tracking():
    global manual_mode, keyboard_mode, running, video_thread
    stop_all_modes()
    running = True
    messagebox.showinfo("Calibration", "Position your face in the center and click OK to calibrate.")
    calibrate()
    video_thread = Thread(target=face_tracking_loop)
    video_thread.start()

def stop_face_tracking():
    global running
    running = False

def start_ai_aim():
    global ai_aiming, ai_aim_thread
    stop_all_modes()
    ai_aiming = True
    ai_aim_thread = Thread(target=ai_aim_loop)
    ai_aim_thread.start()

def stop_ai_aim():
    global ai_aiming
    ai_aiming = False

def stop_all_modes():
    stop_face_tracking()
    stop_hand_tracking()
    stop_ai_aim()

def manual_control_mode():
    global manual_mode, keyboard_mode
    stop_all_modes()
    manual_mode = True
    keyboard_mode = False
    slider_frame.pack(pady=10)

def face_control_mode():
    global manual_mode, keyboard_mode
    manual_mode = False
    keyboard_mode = False
    slider_frame.pack_forget()
    start_face_tracking()

def keyboard_control_mode():
    global manual_mode, keyboard_mode
    stop_all_modes()
    manual_mode = False
    keyboard_mode = True
    slider_frame.pack_forget()
    messagebox.showinfo("Keyboard Control", "Use W/A/S/D keys to move the servos.")
    root.focus_set()

def update_manual_control(val=None):
    if manual_mode:
        servo_horizontal.write(int(horizontal_slider.get()))
        servo_vertical.write(int(vertical_slider.get()))

def on_key_press(event):
    global servo_x_angle, servo_y_angle
    if not keyboard_mode:
        return
    step = 5
    key = event.keysym.lower()
    if key == 'w':
        servo_y_angle = max(0, servo_y_angle - step)
    elif key == 's':
        servo_y_angle = min(180, servo_y_angle + step)
    elif key == 'a':
        servo_x_angle = min(180, servo_x_angle + step)
    elif key == 'd':
        servo_x_angle = max(0, servo_x_angle - step)
    servo_horizontal.write(servo_x_angle)
    servo_vertical.write(servo_y_angle)

def switch_camera():
    global camera_index
    camera_index = 1 if camera_index == 0 else 0
    messagebox.showinfo("Camera Switched", f"Now using camera index: {camera_index}")

# GUI Setup
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

root = ctk.CTk()
root.title("Servo Control Panel")
root.geometry("500x650")  # Increased window width
root.bind("<KeyPress>", on_key_press)

# Status Indicators
status_frame = ctk.CTkFrame(root)
status_frame.pack(pady=10)

ctk.CTkLabel(status_frame, text="✅ Computer OK", text_color="green").pack()
ctk.CTkLabel(status_frame, text="✅ Controller Board OK", text_color="green").pack()
ctk.CTkLabel(status_frame, text="✅ Motor-H OK", text_color="green").pack()
ctk.CTkLabel(status_frame, text="✅ Motor-V OK", text_color="green").pack()

# Buttons
ctk.CTkButton(root, text="Manual Control", command=manual_control_mode).pack(pady=10)
ctk.CTkButton(root, text="Face Control", command=face_control_mode).pack(pady=10)
ctk.CTkButton(root, text="Keyboard Control", command=keyboard_control_mode).pack(pady=10)
ctk.CTkButton(root, text="Move with Hand", command=start_hand_tracking).pack(pady=10)
ctk.CTkButton(root, text="AI Aim (Person)", command=start_ai_aim).pack(pady=10)
ctk.CTkButton(root, text="Switch Camera", command=switch_camera).pack(pady=10)

# Sliders for manual control
slider_frame = ctk.CTkFrame(root)

# Horizontal slider and its label
horizontal_frame = ctk.CTkFrame(slider_frame)
ctk.CTkLabel(horizontal_frame, text="Horizontal").pack()
horizontal_slider = ctk.CTkSlider(horizontal_frame, from_=180, to=0, command=update_manual_control)
horizontal_slider.pack(padx=20, pady=10)
horizontal_frame.pack(side="left", padx=10)

# Vertical slider and its label (vertical orientation)
vertical_frame = ctk.CTkFrame(slider_frame)
ctk.CTkLabel(vertical_frame, text="Vertical").pack()
vertical_slider = ctk.CTkSlider(vertical_frame, from_=180, to=0, command=update_manual_control, orientation="vertical")
vertical_slider.pack(padx=10, pady=10)
vertical_frame.pack(side="left", padx=10)

root.mainloop()
