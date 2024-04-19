# Importing OpenCV Library for basic image processing functions
import cv2
# Numpy for array related functions
import numpy as np
# Dlib for deep learning based Modules and face landmark detection
import dlib
# face_utils for basic operations of conversion
from imutils import face_utils
# alert
# import pygame
import time

class DrowsinessDetection:
    def __init__(self):
        # Initializing the face detector and landmark detector
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

        # status marking for current state
        self.sleep = 0
        self.drowsy = 0
        self.active = 0
        self.status = ""
        self.color = (0, 0, 0)
        self.is_playing = False
        self.start_time = None

        # Initialize pygame mixer
        # pygame.mixer.init()

        # Load the alert sound
        # self.alert_sound = pygame.mixer.Sound("alert.mp3")

    @staticmethod
    def compute_distance(point1, point2):
        dist = np.linalg.norm(point1 - point2)
        return dist

    def detect_eye_status(self, a, b, c, d, e, f):
        short_dist = self.compute_distance(a, b) + self.compute_distance(c, d)
        long_dist = self.compute_distance(e, f)
        ratio = short_dist / (2.0 * long_dist)

        if ratio > 0.25:
            return "Active"
        elif 0.21 < ratio <= 0.25:
            return "Drowsy"
        else:
            return "Sleeping"

    def predict(self, cap):
        out_frame, out_status, out_color = self.detect_drowsiness(cap)
        # cv2.putText(out_frame, out_status, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, out_color, 2)
        # print('status: ', out_status)
        return out_frame, out_status

    def detect_drowsiness(self, cap):
        # _, frame = cap.read()
        frame = cap
        if frame is None:
            return None, None, None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.detector(gray)
        # face_frame = frame.copy()

        if len(faces) == 0:
            self.status = "no face detected"
            return frame, self.status, self.color
            # cv2.putText(frame, self.status, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color, 2)

        else:
            # detected face in faces array
            for face in faces:
                x1 = face.left()
                y1 = face.top()
                x2 = face.right()
                y2 = face.bottom()

                # face_frame = frame.copy()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                landmarks = self.predictor(gray, face)
                landmarks = face_utils.shape_to_np(landmarks)

                # The numbers are actually the landmarks which will show eye
                right_eye_status = self.detect_eye_status(landmarks[43], landmarks[47],
                                                          landmarks[44], landmarks[46],
                                                          landmarks[42], landmarks[45]
                                                          )

                left_eye_status = self.detect_eye_status(landmarks[37], landmarks[41],
                                                         landmarks[38], landmarks[40],
                                                         landmarks[36], landmarks[39]
                                                         )

                # Now judge what to do for the eye
                if right_eye_status == "Sleeping" and left_eye_status == "Sleeping":
                    self.sleep += 1
                    self.drowsy = 0
                    self.active = 0

                    if self.sleep > 6:
                        if self.is_playing and time.time() - self.start_time >= 5:
                            self.is_playing = False

                        if not self.is_playing:
                            self.start_time = time.time()
                            self.is_playing = True
                            # self.alert_sound.play()

                        self.status = "SLEEPING !!!"
                        self.color = (255, 0, 0)

                elif right_eye_status == "Drowsy" and left_eye_status == "Drowsy":
                    self.sleep = 0
                    self.drowsy += 1
                    self.active = 0

                    if self.drowsy > 6:
                        self.status = "Drowsy !"
                        self.color = (0, 0, 255)

                else:
                    self.sleep = 0
                    self.drowsy = 0
                    self.active += 1

                    if self.active > 6:
                        self.is_playing = False
                        # self.alert_sound.stop()
                        self.status = "Active :)"
                        self.color = (0, 255, 0)

                # cv2.putText(frame, self.status, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color, 2)

                for n in range(0, 68):
                    (x, y) = landmarks[n]
                    cv2.circle(frame, (x, y), 1, (255, 255, 255), 5)

        # cv2.imshow("Frame", frame)
        return frame, self.status, self.color
        # cv2.imshow("Result of detector", face_frame)