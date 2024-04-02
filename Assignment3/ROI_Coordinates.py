# For preprocessing raw input images [CROPPING ROI]

import os
import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

def process_ROI():
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(static_image_mode=True, 
                        max_num_hands=2,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)

    dataset_folders = [
        "./dataset/closed/train",
        "./dataset/closed/valid",
        "./dataset/open/train",
        "./dataset/open/valid"
    ]


    def increase_bbox(bbox, scale_factor):
        x, y, w, h = bbox
        delta_w = int((scale_factor - 1) * w / 2)
        delta_h = int((scale_factor - 1) * h / 2)
        return x - delta_w, y - delta_h, w + 2 * delta_w, h + 2 * delta_h



    for dataset_folder in dataset_folders:
        output_folder = dataset_folder.replace("dataset", "dataset").replace("train", "train_ROI").replace("valid", "valid_ROI")
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            
            print(f"[LOG] Extracting ROI for {dataset_folder} folder...")
            for filename in tqdm(os.listdir(dataset_folder)):
                if filename.endswith(".jpg"):
                    img_path = os.path.join(dataset_folder, filename)
                    img = cv2.imread(img_path)
                    img = cv2.flip(img, 1)
                    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                
                    results = hands.process(imgRGB)

                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            landmark_points = np.array([[int(landmark.x * img.shape[1]), int(landmark.y * img.shape[0])] for landmark in hand_landmarks.landmark])
                            x, y, w, h = cv2.boundingRect(landmark_points)
                            x, y, w, h = increase_bbox((x, y, w, h), 1.3)

                            x = max(0, min(x, img.shape[1] - 1))
                            y = max(0, min(y, img.shape[0] - 1))
                            w = max(1, min(w, img.shape[1] - x))
                            h = max(1, min(h, img.shape[0] - y))
                            
                            cropped_img = img[y:y+h, x:x+w]
                            if cropped_img.size == 0:
                                continue

                            output_path = os.path.join(output_folder, filename)
                            cv2.imwrite(output_path, cropped_img)

        else:
            print("[LOG] Skipped ROI extraction from images as it already exists!")
            break

    print("[SUCCESS] Hand detection with cropped region(ROI) inside bounding box saved in the corresponding output folder.")