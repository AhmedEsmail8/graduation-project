import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
from keras.models import load_model
from datetime import datetime
import dlib

class EmotionRecognizer:
    def __init__(self, model_path="models/emotionrecognition.h5"):
        self.model = None
        self.model_path = model_path
        self.emotion_dict = {0: 'anger', 1: 'contempt', 2: 'disgust', 3: 'fear', 4: 'happy', 5: 'sadness', 6: 'surprise'}

    def load_emotion_model(self, model_path):
        model = load_model(model_path)
        return model

    def crop_face(self, image):
        detector = dlib.get_frontal_face_detector()
        faces = detector(image)
        if len(faces) > 0:
            x1 = faces[0].left()
            y1 = faces[0].top()
            x2 = faces[0].right()
            y2 = faces[0].bottom()
            face_roi = image[y1:y2, x1:x2]
            return face_roi, x1, y1, x2, y2
        else:
            return None, None, None, None, None

    def convert_image(self, image):
        pic = cv2.resize(image, (48, 48))
        pic_rgb = cv2.cvtColor(pic, cv2.COLOR_GRAY2RGB)
        image_arr = np.expand_dims(pic_rgb, axis=0)
        image_arr = image_arr.astype('float32') / 255
        prediction = self.model.predict(image_arr)[0]
        ans = np.argmax(prediction)
        return ans, prediction

    def predict(self, image):
        frame, emotion = self.process_frame(image)
        return frame, emotion

    def process_frame(self, frame):
        face_roi, x, y, w, h = self.crop_face(frame)

        if face_roi is not None:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            prediction, probs = self.convert_image(gray)
            emotion = self.emotion_dict[prediction]
            color = (0, 0, 255) if emotion == 'anger' else (255, 0, 0)
            cv2.rectangle(frame, (x, y), (w, h), color, 2)
            cv2.putText(frame, emotion, (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            return frame, emotion
        else:
            return frame, None

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_path = 'output_video.avi'
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        max_frames = 1000
        emotion_data = []

        frame_count = 0
        while cap.isOpened() and frame_count < max_frames:
            time_rec = datetime.now()
            ret, frame = cap.read()
            if ret:
                processed_frame, emotion = self.process_frame(frame)
                out.write(processed_frame)
                if emotion:
                    time_now = datetime.now()
                    emotion_data.append({'Time': time_now, 'Emotion': emotion})
                frame_count += 1
            else:
                break

        cap.release()
        out.release()

        emotion_data_df = pd.DataFrame(emotion_data)
        return emotion_data_df


# # Main function
# def main():
#     model_path = 'models/emotionrecognition.h5'
#     model = load_emotion_model(model_path)
#     frame = cv2.imread("test images/emotion/image0000228.jpg")
#     # video_path = '/kaggle/input/videotest/videoplayback.mp4'
#     emotion_dict = {0: 'anger', 1: 'contempt', 2: 'disgust', 3: 'fear', 4: 'happy', 5: 'sadness', 6: 'surprise'}
#     processed_frame, emotion = process_frame(frame, model, emotion_dict)
#
#     print(emotion)
#     plt.imshow(processed_frame)
#     plt.show()
#
#     # emotion_data_df = process_video(video_path, model, emotion_dict)
#
#     # print(emotion_data_df.head())
#     # print(emotion_data_df.shape)
#
#
#
# main()