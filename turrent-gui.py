import tkinter as tk
from tkinter import ttk, messagebox
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

# Eye landmark indexes
RIGHT_EYE_INDEX = [33, 133]
RIGHT_EYE_TOP_BOTTOM = [159, 145]

neutral_eye_x = None
neutral_eye_y = None
x_range = 120
y_range = 80
manual_mode = True
keyboard_mode = False
video_thread = None
running = False
servo_x_angle = 90
servo_y_angle = 90

def euclidean_distance(pt1, pt2):
    return np.linalg.norm(np.array(pt1) - np.array(pt2))

def calibrate():
    global neutral_eye_x, neutral_eye_y
    cap = cv2.VideoCapture(1)
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
    cap = cv2.VideoCapture(1)
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

def start_face_tracking():
    global manual_mode, keyboard_mode, running, video_thread
    manual_mode = False
    keyboard_mode = False
    running = True
    messagebox.showinfo("Calibration", "Position your face in the center and click OK to calibrate.")
    calibrate()
    video_thread = Thread(target=face_tracking_loop)
    video_thread.start()

def stop_face_tracking():
    global running
    running = False

def manual_control_mode():
    global manual_mode, keyboard_mode
    manual_mode = True
    keyboard_mode = False
    stop_face_tracking()
    slider_frame.pack()

def face_control_mode():
    global manual_mode, keyboard_mode
    manual_mode = False
    keyboard_mode = False
    slider_frame.pack_forget()
    start_face_tracking()

def keyboard_control_mode():
    global manual_mode, keyboard_mode
    manual_mode = False
    keyboard_mode = True
    stop_face_tracking()
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

# GUI Setup
root = tk.Tk()
root.title("Face/Manual/Keyboard Servo Control")
root.geometry("400x350")
root.configure(bg="black")

style = ttk.Style()
style.theme_use("default")
style.configure("TButton", background="skyblue", foreground="black", font=('Helvetica', 10, 'bold'))
style.map("TButton", background=[("active", "deepskyblue")])

btn_manual = ttk.Button(root, text="Manual Control", command=manual_control_mode)
btn_manual.pack(pady=10)

btn_face = ttk.Button(root, text="Face Control", command=face_control_mode)
btn_face.pack(pady=10)

btn_keyboard = ttk.Button(root, text="Keyboard Control", command=keyboard_control_mode)
btn_keyboard.pack(pady=10)

slider_frame = tk.Frame(root, bg="black")

horizontal_slider = tk.Scale(slider_frame, from_=180, to=0, orient=tk.HORIZONTAL, label="Horizontal",
                             command=update_manual_control, bg="black", fg="black",
                             highlightbackground="black", troughcolor="skyblue")
horizontal_slider.pack()

vertical_slider = tk.Scale(slider_frame, from_=0, to=180, orient=tk.VERTICAL, label="Vertical",
                           command=update_manual_control, bg="black", fg="black",
                           highlightbackground="black", troughcolor="skyblue")
vertical_slider.pack()

root.bind("<KeyPress>", on_key_press)
root.protocol("WM_DELETE_WINDOW", lambda: [stop_face_tracking(), root.destroy()])
root.mainloop()
