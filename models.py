# # Importing OpenCV Library for basic image processing functions
# import cv2
# # Numpy for array related functions
# import matplotlib.pyplot as plt
# import numpy as np
# # Dlib for deep learning based Modules and face landmark detection
# import dlib
# # face_utils for basic operations of conversion
# from imutils import face_utils
# # alert
# import pygame
# import time
#
# from DrowsinessDetection import DrowsinessDetection
# from DistractionDetection import DriverDistractionClassifier
# from EmotionDetection import EmotionRecognizer
# from DepthEstimation import DepthEstimator
#
#
# dd = DrowsinessDetection()
# # Initializing the camera and taking the instance
# capture = cv2.VideoCapture('test images/drowsiness/video_1.mp4')
#
# while True:
#     out_frame, out_status, out_color = dd.detect_drowsiness(capture)
#
#     if out_frame is None:
#         break
#
#     cv2.putText(out_frame, out_status, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, out_color, 2)
#     cv2.imshow("Frame", out_frame)
#
#     key = cv2.waitKey(1)
#     if key == 27:
#         break
#
# classifier = DriverDistractionClassifier()
# image = cv2.imread('test images/driver distraction/img_100088.jpg', 0)
# predicted_class, predicted_activity = classifier.predict_class(image)
# print(f"Predicted class: {predicted_class}, Predicted activity: {predicted_activity}")
#
# recognizer = EmotionRecognizer()
# frame = cv2.imread("test images/emotion/image0000228.jpg")
# processed_frame, emotion = recognizer.process_frame(frame)
# plt.imshow(processed_frame)
# plt.show()
# print(emotion)
#
#
#
# depth = DepthEstimator()
# path = "test images/depth estimation/__results___5_0.png"
# image = cv2.imread(path)
# start_time = time.time()
# detection, danger = depth.process_frame(image)
# end_time = time.time()
#
# # Calculate the prediction time
# prediction_time = end_time - start_time
# print("Prediction Time:", prediction_time, "seconds")
#
#
#
# detection = cv2.cvtColor(detection, cv2.COLOR_RGB2BGR)
# plt.imshow(detection)
# plt.axis('off')
# plt.show()
#


import cv2
import matplotlib.pyplot as plt
import numpy as np
import dlib
from imutils import face_utils
import pygame
import time
from multiprocessing import Process, Queue
from DrowsinessDetection import DrowsinessDetection
from DistractionDetection import DriverDistractionClassifier
from EmotionDetection import EmotionRecognizer
from DepthEstimation import DepthEstimator
import torch

# Function to run drowsiness detection
def run_drowsiness_detection():
    capture = cv2.VideoCapture('test images/drowsiness/video_1.mp4')
    dd = DrowsinessDetection()
    while True:
        out_frame, out_status, out_color = dd.detect_drowsiness(capture)
        if out_frame is None:
            break
        cv2.putText(out_frame, out_status, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, out_color, 2)
        cv2.imshow("Frame", out_frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    cv2.destroyAllWindows()

# Function to run distraction detection
def run_distraction_detection():
    classifier = DriverDistractionClassifier()
    image = cv2.imread('test images/driver distraction/img_100088.jpg', 0)
    predicted_class, predicted_activity = classifier.predict_class(image)
    print(f"Predicted class: {predicted_class}, Predicted activity: {predicted_activity}")

# Function to run emotion detection
def run_emotion_detection():
    recognizer = EmotionRecognizer()
    frame = cv2.imread("test images/emotion/image0000228.jpg")
    processed_frame, emotion = recognizer.process_frame(frame)
    plt.imshow(processed_frame)
    plt.show()
    print(emotion)

# Function to run depth estimation
def run_depth_estimation():
    depth = DepthEstimator()
    path = "test images/depth estimation/__results___5_0.png"
    image = cv2.imread(path)
    start_time = time.time()
    detection, danger = depth.process_frame(image)
    end_time = time.time()
    prediction_time = end_time - start_time
    print("Prediction Time:", prediction_time, "seconds")
    detection = cv2.cvtColor(detection, cv2.COLOR_RGB2BGR)
    plt.imshow(detection)
    plt.axis('off')
    plt.show()

import os




if __name__ == '__main__':

    # Get the number of CPU cores
    num_cpu_cores = os.cpu_count()
    print("Number of CPU cores:", num_cpu_cores)

    # Create processes for each prediction task
    processes = []
    processes.append(Process(target=run_drowsiness_detection))
    processes.append(Process(target=run_distraction_detection))
    processes.append(Process(target=run_emotion_detection))
    processes.append(Process(target=run_depth_estimation))

    # Start all processes
    for p in processes:
        p.start()

    # Wait for all processes to finish
    for p in processes:
        p.join()
