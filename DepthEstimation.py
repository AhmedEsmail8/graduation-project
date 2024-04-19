import numpy as np
import cv2
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO
import matplotlib.pyplot as plt
import torch
from keras.models import load_model
from multiprocessing import Process, Queue
import time


class DepthEstimator:
    def __init__(self, yolo_model_path="yolov8n.pt", midas_model_type="DPT_Hybrid", threshold=1221):
        self.yolo_model = None
        self.yolo_model_path = yolo_model_path
        self.midas_model_type = midas_model_type
        self.midas_model = None
        self.threshold = threshold

    def get_mask(self, img):
        # Get image dimensions
        height, width, _ = img.shape

        # Define the four corner points of the polygon (order: top-left, top-right, bottom-right, bottom-left)
        roi_points = np.array([
            [int(width * 0.2), int(height)],
            [int(width * 0.4), int(height * 0.6)],
            [int(width * 0.6), int(height * 0.6)],
            [int(width * 0.8), int(height)]
        ], dtype=np.int32)

        # Create a mask and fill the ROI with white color
        mask = np.zeros_like(img)
        cv2.fillPoly(mask, [roi_points], color=(255, 255, 255))

        # Apply the mask to the image
        masked_img = cv2.bitwise_and(img, mask)

        # Convert the masked image to RGB
        # masked_img_rgb = cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB)

        # Display the masked image
        #     plt.imshow(masked_img_rgb)
        #     plt.axis('off')
        #     plt.show()

        return masked_img

    def get_depth_map(self, img):
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert the image to a PIL Image
        img_pil = Image.fromarray(img2)

        # Define the transformation for MiDaS model
        transform = transforms.Compose([
            transforms.Resize((384, 384)),  # MiDaS model requires input images of size 384x384
            transforms.ToTensor(),
        ])

        # Apply the transformation to the input image
        input_batch = transform(img_pil).unsqueeze(0)

        # Perform depth prediction using MiDaS
        with torch.no_grad():
            prediction = self.midas_model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        # Convert the depth map tensor to a NumPy array for plotting
        depth_map = prediction.cpu().numpy()

        return depth_map

    def predict(self, img):
        img, danger = self.process_frame(img)
        return img, danger

    def process_frame(self, img):
        # Get image dimensions
        start_time = time.time()
        height, width, _ = img.shape  # Extracting height and width
        danger = False

        # Define the four corner points of the polygon (order: top-left, top-right, bottom-right, bottom-left)
        roi_points = np.array([
            [int(width * 0.2), int(height)],
            [int(width * 0.4), int(height * 0.6)],
            [int(width * 0.6), int(height * 0.6)],
            [int(width * 0.8), int(height)]
        ], dtype=np.int32)

        masked_img = self.get_mask(img)
        end_time = time.time()
        print("get mask time: ", end_time - start_time, " seconds")

        # Predict using YOLOv8
        start_time = time.time()
        # results = self.yolo_model(masked_img)
        results = self.yolo_model.predict(masked_img, classes=[0, 1, 2, 3, 5, 7, 9, 10])
        # Display the results
        detect_img = results[0].plot()
        detect_img = cv2.cvtColor(detect_img, cv2.COLOR_BGR2RGB)
        end_time = time.time()
        print("yolo model time: ", end_time - start_time, " seconds")

        #     plt.imshow(detect_img)
        #     plt.axis('off')
        #     plt.show()
        start_time = time.time()
        depth_map = self.get_depth_map(img)
        end_time = time.time()
        print("depth map time: ", end_time - start_time, " seconds")

        start_time = time.time()
        # Assuming you have detected objects using YOLO and obtained their bounding boxes
        detected_objects = results[0].boxes.xyxy  # List of detected objects with bounding boxes

        # Convert image from RGB to BGR color space
        #     image_bgr = cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR)

        # Draw a filled polygon as the ROI on the image
        cv2.polylines(img, [roi_points], isClosed=True, color=(0, 255, 0), thickness=2)  # Change color to green

        for obj in detected_objects:
            x1, y1, x2, y2 = obj  # Extract bounding box coordinates
            object_region_depths = depth_map[int(y1):int(y2), int(x1):int(x2)]  # Extract region from depth map
            # Calculate depth information for the object region
            depth_value = np.max(object_region_depths)  # Example: Compute mean depth

            if depth_value > self.threshold:
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                depth_text = f"{depth_value:.2f}"
                cv2.putText(img, depth_text, (int(x1), int(y2 + 15)), font, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
                danger = True

            else:
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
        end_time = time.time()
        print("drow boxes time: ", end_time - start_time, " seconds")
        return img, danger

# model = YOLO("yolov8m.pt")
#
# # model_type = "DPT_Large"
# model_type = "DPT_Hybrid"
# midas = torch.hub.load("intel-isl/MiDaS", model_type)













