import glob
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import mediapipe as mp
from numpy.core.fromnumeric import resize
from functions import *
from sklearn.preprocessing import StandardScaler

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=True,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.6,
    smooth_landmarks=True
)

#Import the standard scaler
scaler = StandardScaler()
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

body_part_name = []
final_array = []
body_part_iteration = 1
length_array=1
num_non_detect = 0

image_folder = glob.glob("Dataset CAUCAFall/CAUCAFall/*")


for k in range(1):
    image_folder1 = glob.glob(image_folder[k] + "/*")
    for j in range(1):
        image_path = glob.glob(image_folder1[j] + "/*.png")
        for i in range(10):
            image_bgr = cv2.imread(image_path[i])
            # Convert the BGR image (OpenCV format) to RGB for MediaPipe
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            coords = []
            class_fall=[]
            boundary_box = glob.glob(image_folder1[j] + "/*.txt")
            if "classes.txt" in boundary_box[i]:
                continue
            arr = np.loadtxt(boundary_box[i], dtype=float)
            class_fall.append(arr[0])
            x, y, box_width, box_height = arr[1:5].astype(float)
            # Image dimensions to be used for boundary box
            height, width, _ = image_bgr.shape
            x = int(x*width)
            y = int(y*height)
            box_width = int(box_width*width)
            box_height = int(box_height*height)
            # Draw the ROI
            x1 = int(x-box_width/2)
            x2 = int(x+box_width/2)
            y1 = int(y-box_height/2)
            y2 = int(y+box_height/2)
            roi_bgr = image_bgr[y1:y2, x1:x2]
            roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
            results = pose.process(roi_rgb)

            if results.pose_landmarks:
                # Draw the pose annotations on the original BGR image
                annotated_image = roi_bgr.copy()
                mp_drawing.draw_landmarks(
                    annotated_image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )

                landmarks = results.pose_landmarks.landmark
                # For example, get the coordinates of the nose
                for landmark_point in mp_pose.PoseLandmark:
                    lm_value = landmark_point.value
                    lm_name = landmark_point.name
                    if body_part_iteration < 2:
                        body_part_name.append(lm_name + "_x")
                        body_part_name.append(lm_name + "_y")
                        body_part_name.append(lm_name + "_z")
                    coords.append(landmarks[landmark_point].x)
                    coords.append(landmarks[landmark_point].y)
                    coords.append(landmarks[landmark_point].z)

                # Torso Angle, COM, calculation
                torso_angle = calculate_torso_angle(landmarks).ravel()
                com = calculate_com_displacement(landmarks).ravel()
                knee_hip_angle = calculate_knee_hip_angle(landmarks).ravel()
                knee_hip_displacement = calculate_knee_hip_displacement(landmarks).ravel()
                body_part_iteration=body_part_iteration+1

                coords = np.append(coords, [torso_angle, com, knee_hip_angle, knee_hip_displacement, class_fall])

                final_array = np.append(final_array, [coords])
                length_array = length_array+1

                #cv2.imshow("Pose Estimation", resized_annotated_image)
                cv2.imwrite(os.path.join('../Pose_photos_CAUCAFall_ROI', os.path.basename(image_path[i])), annotated_image)
                cv2.waitKey(1)
                cv2.destroyAllWindows()

            else:
                print("No person/pose detected in the image: ", image_path[i])
                num_non_detect = num_non_detect + 1

print(len(final_array))
print(len(body_part_name))
body_part_name = np.append(body_part_name, ("Torso Angle in Degrees", "Center of Mass", "Knee-Hip Angle in Degrees", "Knee-Hip Displacement", "Fall/NoFALL(1,0)"))
final_array = final_array.reshape((length_array-1),104)
scaler.fit(final_array[:,:-1])
final_array1 = scaler.transform(final_array[:,:-1])
print(final_array1)
with open("scaler_m2.pkl", "wb") as f:
    pickle.dump(scaler, f)
final_array = np.append(final_array1, final_array[:,-1]).reshape(((length_array-1),2))
print(final_array)
print(f'The number of photos that were not detected are {num_non_detect}')

df = pd.DataFrame(final_array, columns=body_part_name)  # optionally specify column names
df.to_csv("output_m2_final.csv", index=False)

