import tkinter as tk
from tkinter import messagebox
from my_modules import ImageCapture, ImageDisplay, ModelPrediction, MeasurementLoader, ImageRecapture

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.master.geometry("500x500")  # Set the size of the GUI
        self.pack()
        self.image_capture = ImageCapture()
        self.model_prediction = ModelPrediction("C:/Users/Administrator/Documents/JOBSICENSE/Mywork Project/body_measurement_estimator.keras")
        self.measurements, self.measurement_names = MeasurementLoader.load_measurements("C:/Users/Administrator/Downloads/Compressed/Task 2 bodymesurment/archive/body.csv")
        self.image_recapture = ImageRecapture(self.image_capture)
        self.create_widgets()  # Move this line to the end of the __init__ method

    def create_widgets(self):
        self.instructions_label = tk.Label(self, text="BODY MEASUREMENTS", font=("Arial", 16))
        self.instructions_label.pack(side="top")

        self.instructions_label = tk.Label(self, text="By Joseph Wathome MachineLearning", font=("Arial", 16))
        self.instructions_label.pack(side="bottom")

        self.capture_button = tk.Button(self, text="Capture Images", font=("Arial", 14))
        self.capture_button["command"] = self.capture_images
        self.capture_button.pack(side="top")

        self.display_button = tk.Button(self, text="Display Images", font=("Arial", 14))
        self.display_button["command"] = self.display_images
        self.display_button.pack(side="top")

        self.predict_button = tk.Button(self, text="Predict Measurements", font=("Arial", 14))
        self.predict_button["command"] = self.predict_measurements
        self.predict_button.pack(side="top")

        self.quit = tk.Button(self, text="QUIT", fg="red", command=self.master.destroy, font=("Arial", 14))
        self.quit.pack(side="bottom")

        # In your create_widgets method, add a new button for recapturing images
        self.recapture_button = tk.Button(self, text="Recapture Images", font=("Arial", 14))
        self.recapture_button["command"] = self.recapture_images
        self.recapture_button.pack(side="top")

    def capture_images(self):
        self.image_capture.capture_images()
        messagebox.showinfo("Info", "Images captured!")

    def display_images(self):
        ImageDisplay.display_images(self.image_capture.images)
        messagebox.showinfo("Info", "Images displayed!")

    def predict_measurements(self):
        prediction, predicted_measurements = self.model_prediction.predict(self.image_capture.images, self.measurement_names)
        messagebox.showinfo("Info", f"Measurements predicted: {predicted_measurements}")

    def recapture_images(self):
        self.image_recapture.recapture_images()
        messagebox.showinfo("Info", "Images recaptured!")