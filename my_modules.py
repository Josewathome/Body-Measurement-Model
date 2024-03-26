import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from mediapipe.python.solutions import selfie_segmentation as selfieseg
from tensorflow.keras.models import load_model
import json

class ImageCapture:
    def __init__(self):
        self.cap = None
        self.mp_selfie_segmentation = selfieseg.SelfieSegmentation(model_selection=0)
        self.instructions = {
            'selfie': "Please take a selfie. Look directly at the camera.",
            'front': "Please stand in front of the camera with your full body visible.",
            'side': "Please stand sideways to the camera with your full body visible."
        }
        self.images = {}

    def capture_images(self):
        self.cap = cv2.VideoCapture(0)
        for view in ['selfie', 'front', 'side']:
            print(self.instructions[view])
            while True:
                ret, frame = self.cap.read()
                cv2.imshow('Press Space to Capture', frame)
                if cv2.waitKey(1) & 0xFF == ord(' '):
                    result = self.mp_selfie_segmentation.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    mask = np.stack((result.segmentation_mask,) * 3, axis=-1) > 0.1
                    mask = mask.astype(frame.dtype)
                    segmented_image = np.where(mask, frame, 0)
                    cv2.imwrite(f'segmented_{view}.jpg', segmented_image)
                    self.images[view] = segmented_image 
                    break
        self.cap.release()
        cv2.destroyAllWindows()

class ImageDisplay:
    @staticmethod
    def display_images(images):
        plt.figure(figsize=(10, 5))
        for i, (view, image) in enumerate(images.items(), start=1):
            plt.subplot(1, len(images), i)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title(view)
            plt.axis('off')
        plt.show()

class ModelPrediction:
    def __init__(self, model_path):
        self.model = load_model(model_path)

    @staticmethod
    def preprocess_image(image):
        image = cv2.resize(image, (128, 128))
        return image.astype(np.float32) / 255

    def predict(self, images, measurement_names):
        preprocessed_images = [self.preprocess_image(images[view]) for view in ['selfie', 'front', 'side']]
        images_array = np.array(preprocessed_images, dtype=np.float32)
        prediction = self.model.predict([np.expand_dims(images_array[0], axis=0), np.expand_dims(images_array[1], axis=0), np.expand_dims(images_array[2], axis=0)])
        predicted_measurements = dict(zip(measurement_names[0], prediction[0]))
        print("Predicted measurements:", predicted_measurements)
        return prediction, predicted_measurements

class MeasurementLoader:
    @staticmethod
    def load_measurements(csv_path):
        df = pd.read_csv(csv_path)
        measurements = []
        measurement_names = []
        for index, row in df.iterrows():
            with open(row['measurements'], 'r') as f:
                measurement = json.load(f)
            measurement = {k: v.replace('_tbr', '') if isinstance(v, str) else v for k, v in measurement.items()}
            measurement = {k: float(v) for k, v in measurement.items() if k != 'gender' and k != 'race' and k != 'profession'}
            if 'gender' in measurement:
                measurement['gender'] = 1.0 if measurement['gender'] == 'male' else 0.0
            measurements.append(list(measurement.values()))
            measurement_names.append(list(measurement.keys()))
        return measurements, measurement_names

class ImageRecapture:
    def __init__(self, image_capture):
        self.image_capture = image_capture

    def recapture_images(self):
        self.image_capture.images = {}
        self.image_capture.capture_images()