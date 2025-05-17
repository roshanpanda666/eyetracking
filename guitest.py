import customtkinter as ctk

# Setup CustomTkinter appearance
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class SliderApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Slider Example")
        self.geometry("400x300")

        # Slider 1
        self.slider1 = ctk.CTkSlider(self, from_=0, to=180, number_of_steps=180)
        self.slider1.pack(pady=20, padx=40)

        # Slider 2
        self.slider2 = ctk.CTkSlider(self, from_=0, to=180, number_of_steps=180)
        self.slider2.pack(pady=20, padx=40)

        # Button
        self.print_button = ctk.CTkButton(self, text="Print Slider Values", command=self.print_values)
        self.print_button.pack(pady=20)

    def print_values(self):
        val1 = self.slider1.get()
        val2 = self.slider2.get()
        print(f"Slider 1 value: {val1}")
        print(f"Slider 2 value: {val2}")

if __name__ == "__main__":
    
    app = SliderApp()
    app.mainloop()
    
