from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import shutil
import time
from DrowsinessDetection import DrowsinessDetection
from DistractionDetection import DriverDistractionClassifier
from EmotionDetection import EmotionRecognizer
from DepthEstimation import DepthEstimator
from generate_video import *


app = Flask(__name__, static_folder='static')
dd = DrowsinessDetection()
distraction_model = DriverDistractionClassifier()
emotion_recognizer = EmotionRecognizer()
depth_estimator = DepthEstimator()
# app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024
# app.config['TIMEOUT'] = 60

def delete_folder_contents(folder_path):
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"The folder '{folder_path}' does not exist.")
        return

    # Iterate over all the files and subfolders in the folder
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                # Remove the file
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
            except Exception as e:
                print(f"Error deleting file: {file_path}. Error: {e}")

        for dir in dirs:
            dir_path = os.path.join(root, dir)
            try:
                # Remove the directory and its contents recursively
                shutil.rmtree(dir_path)
                print(f"Deleted folder: {dir_path}")
            except Exception as e:
                print(f"Error deleting folder: {dir_path}. Error: {e}")


@app.route('/')
def uploader():
    return render_template('index.html')


@app.route('/simulation', methods=['POST'])
def simulation():
    if all(filekey in request.files for filekey in
           ['emotion-video', 'drowsiness-video', 'depth-video', 'distraction-video']):
        for filekey in ['emotion-video', 'drowsiness-video', 'depth-video', 'distraction-video']:
            file = request.files[filekey]
            if file.filename == '':
                return 'No selected file'
            file.save(os.path.join('upload', filekey.replace('-', '_')+'.mp4'))

        # start_time = time.time()
        # predict(distraction_video='upload/drowsiness_video.mp4', emotion_video='upload/emotion_video.mp4',
        #         road_video='upload/depth_video.mp4', drowsiness_video='upload/drowsiness_video.mp4', drowsiness_detector=dd,
        #         emotion_recognizer=emotion_recognizer, distraction_model=distraction_model, depth_estimator=depth_estimator,
        #         save_folder='static')
        # end_time = time.time()
        # print("program time:", end_time - start_time)
        #
        # while not os.path.isfile('static/video.mp4'):
        #     print('yes')

        return render_template('simulation.html', video_path='depth_video.mp4')
    else:
        return 'Missing file(s)'

    # Check if the POST request has the file part
    # print(request.files)
    #
    # if 'emotion-video' not in request.files or 'drowsiness-video' not in request.files or 'depth-video' not in request.files or 'distraction-video' not in request.files:
    #     return 'No file part'
    #
    # emotion_video = request.files['emotion-video']
    # drowsiness_video = request.files['drowsiness-video']
    # depth_video = request.files['depth-video']
    # distraction_video = request.files['distraction-video']
    #
    # # If user does not select file, browser also submit an empty part without filename
    # if emotion_video.filename == '' or drowsiness_video.filename == '' or depth_video.filename == '' or distraction_video.filename == '':
    #     return 'No selected file'
    #
    # print(emotion_video)
    # print(drowsiness_video)
    # print(depth_video)
    # print(distraction_video)
    #
    # delete_folder_contents('upload')
    #
    # emotion_video.save('upload/emotion_video.mp4')
    # drowsiness_video.save('upload/drowsiness_video.mp4')
    # depth_video.save('upload/depth_video.mp4')
    # distraction_video.save('upload/distraction_video.mp4')
    #
    # # start_time = time.time()
    # # predict(distraction_video='upload/drowsiness_video.mp4', emotion_video='upload/emotion_video.mp4',
    # #         road_video='upload/depth_video.mp4', drowsiness_video='upload/drowsiness_video.mp4', drowsiness_detector=dd,
    # #         emotion_recognizer=emotion_recognizer, distraction_model=distraction_model, depth_estimator=depth_estimator)
    # # end_time = time.time()
    # # print("program time:", end_time - start_time)
    #
    # # while not os.path.isfile('video.mp4'):
    # #     print('yes')
    #
    # return render_template('simulation.html', video_path='video.mp4')


if __name__ == '__main__':
    app.run(debug=True)
