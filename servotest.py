import time
from pyfirmata import Arduino
import customtkinter as ctk

# Setup appearance
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

# Create window
root = ctk.CTk()
root.geometry("500x400")
root.title("Live Servo Control")

# Connect to Arduino
board = Arduino('COM8')
horizontal = board.get_pin('d:9:s')
vertical = board.get_pin('d:12:s')

# Function to send slider values to servos
def update_horizontal(val):
    print(f"Horizontal Slider: {val}")
    horizontal.write(float(val))

def update_vertical(val):
    print(f"Vertical Slider: {val}")
    vertical.write(float(val))

# Frame to organize sliders side by side
frame = ctk.CTkFrame(root)
frame.pack(pady=20, padx=20, expand=True)

# Horizontal slider (left)
slider1 = ctk.CTkSlider(frame, from_=180, to=0, number_of_steps=180, orientation="horizontal", command=update_horizontal)
slider1.set(90)
slider1.grid(row=0, column=0, padx=20, pady=10)

# Vertical slider (right)
slider2 = ctk.CTkSlider(frame, from_=180, to=0, number_of_steps=180, orientation="vertical", command=update_vertical)
slider2.set(90)
slider2.grid(row=0, column=1, padx=20, pady=10, sticky="ns")

# Run the GUI
root.mainloop()
