import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import mediapipe as mp
from numpy import dtype
from numpy.core.fromnumeric import resize
from sklearn.preprocessing import MinMaxScaler
from functions import calculate_torso_angle, calculate_knee_hip_angle, calculate_knee_hip_displacement, \
    calculate_com_displacement, get_bounding_box, boundary_box_ratio


def initialisation_mp():
    # Real-time Pose Detection with Fall Detection
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=True,  # if True = single image input (no tracking between frames)
        model_complexity=2,  # 0,1,2 - higher number = more accurate but slower
        enable_segmentation=True,  # Set True if you need the segmentation mask
        min_detection_confidence=0.4,
        min_tracking_confidence=0.6,
        smooth_landmarks=True
    )

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    return pose, mp_drawing, mp_drawing_styles, mp_pose


pose, mp_drawing, mp_drawing_styles, mp_pose = initialisation_mp()

body_part_name = []
final_array = []
body_part_iteration = 1
length_array = 1

for j in range(2):
    if j == 0:
        image_path = glob.glob("fall_dataset/images/train/*.jpg")
        image_path_label = glob.glob("fall_dataset/labels/train/*.txt")
    else:
        image_path = glob.glob("fall_dataset/images/val/*.jpg")
        image_path_label = glob.glob("fall_dataset/labels/val/*.txt")
    for i in range(len(image_path)):

        image_bgr = cv2.imread(image_path[i])
        arr = np.genfromtxt(image_path_label[i], delimiter=' ', dtype=float)
        filename = os.path.basename(image_path[i])  # Get filename from full path
        if "fallen023" in filename or "fallen024" in filename:
            print(f"Skipping {filename}")
            continue
        elif arr.shape[0] == 5:
            rows_num = 1
        else:
            rows_num = arr.shape[0]
        for row in range(1):
            coords = []
            class_fall = []
            if rows_num == 1:
                x_roi, y_roi, box_width, box_height = arr[1:5].astype(float)
                if arr[0].astype(float) == 1 or arr[0].astype(float) == 2: class_fall.append(0)
                if arr[0].astype(float) == 0: class_fall.append(1)
            else:
                x_roi, y_roi, box_width, box_height = arr[row, 1:5].astype(float)
                if arr[row, 0].astype(float) == 1 or arr[row, 0].astype(float) == 2: class_fall.append(1)
                if arr[row, 0].astype(float) == 0: class_fall.append(0)
            # Convert the BGR image (OpenCV format) to RGB for MediaPipe
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            # Image dimensions to be used for boundary box
            height, width, _ = image_bgr.shape
            x_roi = int(x_roi * width)
            y_roi = int(y_roi * height)
            box_width = int(box_width * width)
            box_height = int(box_height * height)
            # Draw the ROI
            x1 = int(x_roi - (box_width / 2))
            x2 = int(x_roi + (box_width / 2))
            y1 = int(y_roi - (box_height / 2))
            y2 = int(y_roi + (box_height / 2))
            roi_bgr = image_bgr[y1:y2, x1:x2]
            roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
            # For fall detection classification
            results = pose.process(image_rgb)

            # --- Check if any landmarks were found ---
            if results.pose_landmarks:
                # Draw the pose annotations on the original BGR image
                annotated_image = image_bgr.copy()
                mp_drawing.draw_landmarks(
                    annotated_image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )
                landmarks = results.pose_landmarks.landmark

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

                # Torso Angle, COM, and knee-hip angle calculation
                torso_angle = calculate_torso_angle(landmarks).ravel()
                com = calculate_com_displacement(landmarks).ravel()
                knee_hip_angle = calculate_knee_hip_angle(landmarks).ravel()
                knee_hip_displacement = calculate_knee_hip_displacement(landmarks).ravel()
                box_ratio = boundary_box_ratio(box_width, box_height)
                body_part_iteration = body_part_iteration + 1
                coords = np.append(coords, [torso_angle, com, knee_hip_angle, knee_hip_displacement, class_fall])
                final_array = np.append(final_array, [coords])
                length_array = length_array + 1
                resized_annotated_image = cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                resized_annotated_image = cv2.resize(annotated_image, None, fx=4.0, fy=4.0)


            else:
                print("No person/pose detected in the image: ", image_path[i])


        # cv2.imshow("Pose Estimation", resized_annotated_image)
        cv2.imwrite(os.path.join('../Pose_photos_m1_roi', os.path.basename(image_path[i])), resized_annotated_image)
        cv2.waitKey(1)
        cv2.destroyAllWindows()

print(len(final_array))
print(len(body_part_name))
body_part_name = np.append(body_part_name, (
"Torso Angle in Degrees", "Center of Mass", "Knee-Hip Angle in Degrees", "Knee-Hip Displacement", "Fall/NoFALL(1,0)"))

final_array = final_array.reshape((length_array - 1), 104)
print(final_array)

df = pd.DataFrame(final_array, columns=body_part_name)
df.to_csv("output_m1.csv", index=False)
