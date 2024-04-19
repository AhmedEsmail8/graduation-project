import cv2
import numpy as np


class DriverDistractionClassifier:
    def __init__(self, model_path='models/driver_distraction_model.h5', img_rows=64, img_cols=64, color_type=1):
        self.model = None
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.color_type = color_type
        self.model_path = model_path
        self.activity_map = {'c0': 'Safe driving',
                             'c1': 'Texting - right',
                             'c2': 'Talking on the phone - right',
                             'c3': 'Texting - left',
                             'c4': 'Talking on the phone - left',
                             'c5': 'Operating the radio',
                             'c6': 'Drinking',
                             'c7': 'Reaching behind',
                             'c8': 'Hair and makeup',
                             'c9': 'Talking to passenger'}

    def _preprocess_image(self, img):
        img = cv2.resize(img, (self.img_rows, self.img_cols))
        img = np.array(img, dtype=np.uint8)
        img = img.reshape(-1, self.img_rows, self.img_cols, self.color_type)
        return img

    def predict(self, img):
        predicted_class, predicted_activity = self.predict_class(img)
        return img, predicted_activity

    def predict_class(self, img):
        img_processed = self._preprocess_image(img)
        y_prediction = self.model.predict(img_processed, batch_size=32, verbose=1)
        predicted_class = np.argmax(y_prediction)
        return predicted_class, self.activity_map.get('c{}'.format(predicted_class))


