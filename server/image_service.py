import torch
import cv2
import pandas as pd
import numpy as np
import uuid
import os

def get_blurred_image_path(original_image: str, extension: str):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path = './yolov5_v2.pt')

    results = model(original_image)

    image = cv2.imread(original_image)

    blurred_image = cv2.blur(image,(555, 555))

    for box in results.xyxy[0]: 
        xB = int(box[2])
        xA = int(box[0])
        yB = int(box[3])
        yA = int(box[1])

        w = xB - xA
        h = yA - yB

        roi_corners = np.array([[(xA, yA - h), (xA, yA), (xB, yB + h), (xB, yB)]], dtype=np.int32)

        mask = np.zeros(image.shape, dtype=np.uint8)
        channel_count = image.shape[2]
        ignore_mask_color = (255,)*channel_count
        cv2.fillPoly(mask, roi_corners, ignore_mask_color)

        mask_inverse = np.ones(mask.shape).astype(np.uint8)*255 - mask

        final_image = cv2.bitwise_and(blurred_image, mask) + cv2.bitwise_and(image, mask_inverse)

        temp_result_location = os.path.join("uploaded_images", str(uuid.uuid4()) + extension)

        cv2.imwrite(temp_result_location, final_image)

        return temp_result_location
    
 