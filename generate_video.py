import io
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
import os
import time
from multiprocessing import Process, Queue
import multiprocessing
from keras.models import load_model
from ultralytics import YOLO
import torch
import tkinter as tk
from tkinter import messagebox




def create_collage(images, predictions, speed):


    # images = [Image.open(path) for path in image_paths]
    fig, axs = plt.subplots(2, 2)
    titles = ["Drowsiness Detection", "Distraction Detection", "Emotion Recognition", "Depth estimation"]
    texts = [f"Prediction: {prediction}" for prediction in predictions] + [f"Speed: {speed}"]
    text_colors = ['red'] * len(predictions) + ['blue']

    label = 'Prediction: '
    for i in range(2):
        for j in range(2):
            index = i * 2 + j
            if index < len(images):
                axs[i, j].imshow(images[index][0])
                axs[i, j].axis('off')
                axs[i, j].set_title(titles[index])
                if index == 3:
                    label = "Speed: "
                else:
                    label = 'Prediction: '
                axs[i, j].text(0.5, -0.1, label + str(images[index][1]), color=text_colors[index], horizontalalignment='center', verticalalignment='center', transform=axs[i, j].transAxes)

    plt.tight_layout()

    # Convert plot to image in memory
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    plt.close()

    frame = Image.open(buf)
    return frame


def images_to_video(images, output_video_path, fps=25):
    # Get the size of the first image to determine frame dimensions
    frame_width, frame_height = images[0].size
    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Choose codec (XVID, mp4v, etc.)
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    # Convert each image to numpy array and write to video
    for image in images:
        image_np = np.array(image)
        # OpenCV uses BGR color format, so convert RGB to BGR
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        video_writer.write(image_bgr)
    video_writer.release()
    print(f"Video created successfully: {output_video_path}")


def extract_frames(video_path):
    # Open the video file
    video_capture = cv2.VideoCapture(video_path)

    frames = []

    # Read until video is completed
    while video_capture.isOpened():
        # Read a frame from the video
        ret, frame = video_capture.read()

        # If frame is read correctly
        if ret:
            # Convert the frame from BGR to RGB (if necessary)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Append the frame to the list
            frames.append(frame_rgb)
        else:
            break

    # Release the video capture object
    video_capture.release()

    return frames

def generate_batches(frames1, frames2, frames3, frames4, batch_size = 20):
    min_frames = min(len(frames1), len(frames2), len(frames3), len(frames4))
    lst = [[], [], [], []]
    speeds = []
    for i in range(0, min_frames, batch_size):
        speed = f"{i}km/h"
        tmp = min(min_frames, i+batch_size)
        lst[0].append(frames1[i:tmp])
        lst[1].append(frames2[i:tmp])
        lst[2].append(frames3[i:tmp])
        lst[3].append(frames4[i:tmp])
        speeds.append(speed)

    return speeds, lst


# output_frames = [[], [], [], []]
def collage(distraction_frames, drowsiness_frames, road_frames, emotion_frames, process, return_dict):
    lst = []
    predictions = ["safe", "in danger", "safe"]
    speed = "80km/h"
    for f in range(len(distraction_frames)):

        out_frame = create_collage([distraction_frames[f], drowsiness_frames[f], emotion_frames[f], road_frames[f]], predictions, speed)
        lst.append(out_frame)
    return_dict[process] = lst

def get_prediction_frames(lst, model, return_dict, process_idx, skip=0):

    if process_idx == 1 or process_idx == 3:
        model.model = load_model(model.model_path)
    elif process_idx == 2:
        model.yolo_model = YOLO(model.yolo_model_path)
        model.midas_model = torch.hub.load("intel-isl/MiDaS", model.midas_model_type)


    print(type(model))
    start_time = time.time()
    out_lst = []
    predictions = []
    last_frame = lst[0]
    last_prediction = 'no face detected'
    for i in range(len(lst)):
        frame = lst[i]
        if i%(skip+1)==0:
            last_frame, last_prediction = model.predict(frame)
        if last_prediction=='' and process_idx==0:
            last_prediction='Active :)'
        out_lst.append([last_frame, last_prediction])
        # print(out_lst[-1])
    # lst[:] = out_lst
    return_dict[process_idx] = out_lst
    end_time = time.time()
    print('model idx#', process_idx, 'DONE... in time =', end_time-start_time)
    # return out_lst



def predict(distraction_video, drowsiness_video, road_video, emotion_video, drowsiness_detector, distraction_model, depth_estimator, emotion_recognizer, save_folder):

    if emotion_video==None:
        emotion_video=drowsiness_video

    plots = []

    # Example usage
    start_time = time.time()


    distraction_frames = extract_frames(distraction_video)
    drowsiness_frames = extract_frames(drowsiness_video)
    road_frames = extract_frames(road_video)
    emotion_frames = extract_frames(emotion_video)
    end_time = time.time()
    prediction_time = end_time - start_time
    print("Extracting frames Time:", prediction_time, "seconds")
    i = 0


    start_time = time.time()
    pred_manager = multiprocessing.Manager()
    pred_dict = pred_manager.dict()
    pred_processes = [Process(target=get_prediction_frames, args=(drowsiness_frames, drowsiness_detector, pred_dict, 0, 30)),
                      Process(target=get_prediction_frames, args=(distraction_frames, distraction_model, pred_dict, 1, 30)),
                      Process(target=get_prediction_frames, args=(road_frames, depth_estimator, pred_dict, 2, 30)),
                      Process(target=get_prediction_frames, args=(emotion_frames, emotion_recognizer, pred_dict, 3, 30))]

    for p in pred_processes:
        p.start()

    for p in pred_processes:
        p.join()


    # drowsiness_frames = get_prediction_frames(drowsiness_frames, dd, pred_dict, 0, skip=0)
    # distraction_frames = get_prediction_frames(distraction_frames, dd, pred_dict, 1, skip=0)
    # road_frames = get_prediction_frames(road_frames, dd, pred_dict, 2, skip=0)
    # emotion_frames = get_prediction_frames(emotion_frames, dd, pred_dict, 3, skip=0)
    end_time = time.time()
    print('Predict all models:', end_time - start_time)

    start_time = time.time()
    drowsiness_frames = pred_dict[0]
    distraction_frames = pred_dict[1]
    road_frames = pred_dict[2]
    emotion_frames = pred_dict[3]
    # drowsiness_frames, distraction_frames, road_frames, emotion_frames = pred_dict.values()
    end_time = time.time()
    print('fill prediction frames time: ', end_time-start_time)


    start_time = time.time()
    speeds, batches = generate_batches(drowsiness_frames, distraction_frames, road_frames, emotion_frames, 20)
    num_of_batches = len(batches[0])
    print(num_of_batches)
    num_of_processes = 4
    k = 0

    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    process_idx = 0

    while k < num_of_batches:
        valid_processes = min(num_of_processes, num_of_batches - k)
        print("valid processes : ", valid_processes)
        processes = []
        for j in range(valid_processes):
            processes.append(Process(target=collage, args=(
            batches[0][k + j], batches[1][k + j], batches[2][k + j], batches[3][k + j], process_idx, return_dict)))
            process_idx += 1
        for p in processes:
            p.start()

        # Wait for all processes to finish
        for p in processes:
            p.join()
        k += valid_processes
    end_time = time.time()
    prediction_time = end_time - start_time
    print("creating parallel collage Time:", prediction_time, "seconds")
    for d in return_dict:
        plots.extend(return_dict[d])

    start_time = time.time()

    # plt.imshow(plots[0])
    # plt.show()

    images_to_video(plots, os.path.join(save_folder, 'video.mp4'))

    end_time = time.time()
    prediction_time = end_time - start_time
    print("generate video Time:", prediction_time, "seconds")



# if __name__ == '__main__':
#     from DrowsinessDetection import DrowsinessDetection
#     from DistractionDetection import DriverDistractionClassifier
#     from EmotionDetection import EmotionRecognizer
#     from DepthEstimation import DepthEstimator
#     dd = DrowsinessDetection()
#     distraction_model = DriverDistractionClassifier()
#     emotion_recognizer = EmotionRecognizer()
#     depth_estimator = DepthEstimator()
#     root = tk.Tk()
#     root.withdraw()
#     start_time = time.time()
#     predict("test images/driver distraction/10 sec.mp4", "test images/drowsiness/0401.mp4", "test images/drowsiness/0401.mp4", "test images/drowsiness/0401.mp4")
#     end_time = time.time()
#     print("program time:", end_time-start_time)
#     # root.deiconify()
#     # root.mainloop()