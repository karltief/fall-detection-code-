import cv2
import joblib
import mediapipe as mp
import numpy as np
import time
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import itertools
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler, label_binarize
file_path = r"C:\Users\UOE\PycharmProjects\ML_Fall\best_knn_model_emergency.pkl"
emergency_model = joblib.load(file_path)
start_time = None
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

def initialisation_mp():
    # Real-time Pose Detection with Fall Detection
    mp_pose = mp.solutions.pose

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=True,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.3,
        smooth_landmarks=True
    )

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    return pose, mp_drawing, mp_drawing_styles, mp_pose
pose, mp_drawing, mp_drawing_styles, mp_pose = initialisation_mp()

def plot_c_score(grid, filename, model_name, dataset, x1, x2):
    test_score = grid.cv_results_["mean_test_score"]
    params = [d['C'] for d in grid.cv_results_['params']]
    fig,ax = plt.subplots()
    ax.scatter(params, test_score)
    ax.plot(params, test_score)
    ax.set_xlabel('C')
    ax.set_ylabel('Test Score')
    ax.set_title(f'Accuracy vs C with Zoomed-in Region for {model_name} on Dataset {dataset}')
    plot_inserted = inset_axes(ax, width="30%", height="30%", loc='lower right', borderpad=2)
    plot_inserted.plot(params, test_score)
    y1, y2 = (0.99*np.max(test_score)), np.max(test_score)
    plot_inserted.set_xlim(x1, x2)
    plot_inserted.set_ylim(y1, y2)
    mark_inset(ax, plot_inserted, loc1=2, loc2=4, fc="none", ec="0.5")
    plt.savefig(f"plot_c_vs_test_score_{filename}.png")
    return
# Mutual information to understand which parameter has the most influence on prediction
def mutual_info(X, y):
    mi_series = mutual_info_classif(X, y)
    mi_series = pd.DataFrame({"Feature": X.columns, "MI": mi_series})
    return mi_series.sort_values(by='MI', ascending=False)

def mutual_info_plot(plot):
    plt.figure(figsize=(22, 6))
    sns.barplot(x=plot["Feature"], y=(plot["MI"]), palette="coolwarm")
    plt.xticks(rotation=90, fontsize=13)
    plt.yticks(fontsize=13)
    plt.xlabel("Feature Name", fontsize=18)
    plt.ylabel("Weight (Importance)", fontsize=18)
    plt.title("Feature Importance for Fall Detection", fontsize=22)
    plt.savefig(f"weightsC.png", bbox_inches='tight')
    plt.show(bbox_inches='tight')

# Torso Angle Calculation
def calculate_torso_angle(landmarks):
    left_shoulder = np.array([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
                              landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z])
    right_shoulder = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
                               landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z])
    left_hip = np.array([landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z])
    right_hip = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
                          landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z])

    torso_vector = (left_shoulder + right_shoulder) / 2 - (left_hip + right_hip) / 2
    vertical_vector = np.array([0, -1, 0])  # 3D vertical axis

    angle = np.arccos(np.dot(torso_vector, vertical_vector) / (np.linalg.norm(torso_vector)))
    angle = np.radians(angle)
    return angle

# Center of Mass calculation
def calculate_com_displacement(landmarks):
    left_hip = np.array([landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z])
    right_hip = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
                          landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z])
    left_ankle = np.array([landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,
                           landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].z])
    right_ankle = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y,
                            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].z])

    com = (left_hip + right_hip) / 2
    feet_center = (left_ankle + right_ankle) / 2
    displacement = np.linalg.norm(com - feet_center)
    return displacement


def calculate_knee_hip_angle(landmarks):
    left_knee = np.array([landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y,
                         landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].z])
    right_knee = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y,
                          landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].z])
    left_hip = np.array([landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z])
    right_hip = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
                          landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z])
    left_ankle = np.array([landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,
                           landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].z])
    right_ankle = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y,
                            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].z])

    feet_center = (left_ankle + right_ankle) / 2
    com = (left_hip + right_hip) / 2
    knee = (left_knee + right_knee) / 2
    hip_to_knee = knee - com
    knee_to_ankle = feet_center - knee
    cos_theta = np.dot(hip_to_knee, knee_to_ankle) / (np.linalg.norm(hip_to_knee) * np.linalg.norm(knee_to_ankle))
    angle = np.degrees(np.arccos(cos_theta))

    return angle

def calculate_knee_hip_displacement(landmarks):
    left_hip = np.array([landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].z])
    right_hip = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
                          landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].z])
    left_knee = np.array([landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y,
                           landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].z])
    right_knee = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y,
                            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].z])

    com = (left_hip + right_hip) / 2
    knee_center = (left_knee + right_knee) / 2
    displacement = np.linalg.norm(com - knee_center)
    return displacement
def hand_distance(landmarks):
    right_pinky = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_PINKY.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_PINKY.value].y,
                         landmarks[mp_pose.PoseLandmark.RIGHT_PINKY.value].z])
    right_thumb = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_THUMB.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_THUMB.value].y,
                          landmarks[mp_pose.PoseLandmark.RIGHT_THUMB.value].z])
    left_pinky = np.array([landmarks[mp_pose.PoseLandmark.LEFT_PINKY.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_PINKY.value].y,
                           landmarks[mp_pose.PoseLandmark.LEFT_PINKY.value].z])
    left_thumb = np.array([landmarks[mp_pose.PoseLandmark.LEFT_THUMB.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_THUMB.value].y,
                            landmarks[mp_pose.PoseLandmark.LEFT_THUMB.value].z])

    left_hand = (left_pinky + left_thumb) / 2
    right_hand = (right_pinky + right_thumb) / 2
    scaler = StandardScaler()
    displacement = np.linalg.norm(left_hand - right_hand)
    return displacement
def opp_hand_elbow_angle(landmarks):
    right_elbow = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,
                         landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].z])
    right_wrist = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y,
                          landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].z])
    left_elbow = np.array([landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
                           landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].z])
    left_wrist = np.array([landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,
                            landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].z])

    cos_theta_left = np.dot(right_elbow, left_wrist) / (np.linalg.norm(right_elbow) * np.linalg.norm(left_wrist))
    angle1 = np.degrees(np.arccos(cos_theta_left))
    cos_theta_right = np.dot(left_elbow, right_wrist) / (np.linalg.norm(left_elbow) * np.linalg.norm(right_wrist))
    angle2 = np.degrees(np.arccos(cos_theta_right))
    ratio = angle1/angle2
    return ratio
def get_bounding_box(landmarks, frame_width, frame_height):

    x_min, y_min = frame_width, frame_height
    x_max, y_max = 0, 0

    for landmark in landmarks:
        x, y = int(landmark.x * frame_width), int(landmark.y * frame_height)
        if x < x_min: x_min = x
        if y < y_min: y_min = y
        if x > x_max: x_max = x
        if y > y_max: y_max = y

        box_width = x_max - x_min
        box_height = y_max - y_min
        padding_factor = 0.01
        x_min = max(0, int(x_min - padding_factor * box_width))
        y_min = max(0, int(y_min - padding_factor * box_height))
        x_max = min(frame_width, int(x_max + padding_factor * box_width))
        y_max = min(frame_height, int(y_max + padding_factor * box_height))

    return x_min, y_min, x_max, y_max

def boundary_box_ratio(box_width, box_height):
    return box_width / box_height

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.figure(figsize=(8,6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title,fontsize=18)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45,fontsize=12)
    plt.yticks(tick_marks, classes,fontsize=12)
    fmt = '.3f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center", fontsize=16,
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label', fontsize=18)
    plt.xlabel('Predicted label', fontsize =18)
    plt.savefig(f"{title}.png", bbox_inches='tight')
    plt.show()
    return


def shoulder_width(landmarks):
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    # Shoulder width for scaling
    shoulder_width = np.sqrt((left_shoulder.x - right_shoulder.x) ** 2 + (left_shoulder.y - right_shoulder.y) ** 2)
    return shoulder_width

alert = 0
previous_ratio = None
head_state = None
shake_number = 3
shake_counter = 0
previous_yaw = None
last_shake_time = None
yaw_array = []
trigger = 1
def emergency_gesture_rec(coords, landmarks):
    global start_time, alert, shoulder_width, angular_velocity
    '''
    coords_emergency = []
    pose_array_emergency = []
    hand_distance_d = hand_distance(landmarks).ravel()
    coords_emergency = np.append(coords, [hand_distance_d])
    pose_array_emergency = np.append(pose_array_emergency, [coords_emergency]).reshape(-1, 100)
    emergency_prediction_from_model = emergency_model.predict(pose_array_emergency)[0]
    '''
    # Hardcode positions
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
    # Distance scaler using shoulder width
    shoulder_width_value = shoulder_width(landmarks)
    # Opposite wrist-shoulder distance
    distance_rightS_leftW = np.sqrt((left_wrist.x-right_shoulder.x)**2+(left_wrist.y-right_shoulder.y)**2)
    distance_leftS_rightW = np.sqrt((right_wrist.x-left_shoulder.x)**2+(right_wrist.y-left_shoulder.y)**2)
#  and left_elbow.x < left_shoulder.x and right_elbow.x > right_shoulder.x\
    scale_threshold = shoulder_width_value * 0.3
    z_threshold = shoulder_width_value * 2.0
    distance_condition = abs(left_wrist.z - right_shoulder.z) < z_threshold and abs(right_wrist.z - left_shoulder.z) < z_threshold
    if (distance_rightS_leftW < scale_threshold) and (distance_leftS_rightW < scale_threshold):# and distance_condition:
        if start_time is None:
            start_time = time.time()
        time_elapsed = time.time() - start_time
        if time_elapsed >=2:
            alert = 1
    else:
        start_time = None
    emergency_prediction = alert
    if alert == 1:
       alert, ang_v = headshaking(landmarks)
    return emergency_prediction, shoulder_width_value
def calculate_yaw_angle(landmarks):
    left_shoulder = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]
    nose = landmarks[mp.solutions.pose.PoseLandmark.NOSE.value]
    # Compute yaw angle using shoulder & nose position
    yaw_angle = np.arctan2((nose.x - (left_shoulder.x + right_shoulder.x) / 2)/ shoulder_width(landmarks), 1) * (180 / np.pi) # Scaling factor
    return yaw_angle

def angular_velocity_calc(landmarks):
    global yaw_array
    angular_v = 0
    # Yaw angle
    yaw_angle = calculate_yaw_angle(landmarks)
    yaw_array.append(yaw_angle)
    if len(yaw_array) > 5:  # Keep the last 5 yaw values
        yaw_array.pop(0)
    if len(yaw_array) >= 2:
        # To get the angular velocity in degrees per second multiply the equation below by the frame rate of the camera. Right now it is degrees per frame
        angular_v = abs(yaw_array[-1] - yaw_array[0]) / len(yaw_array)
    return angular_v, yaw_angle
angular_velocity_threshold = 2.2
def headshaking(landmarks):
    global previous_ratio, head_state, shake_number, start_time, shake_counter, previous_yaw, last_shake_time, trigger
    detected_shake = False
    trigger = 1
    shake_threshold = shoulder_width(landmarks) * 0.19
    # Nose used for headshaking
    nose_x = landmarks[mp_pose.PoseLandmark.NOSE.value].x
    left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
    right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value]

    # Compute ratio of left ear-to-nose and right ear-to-nose distances
    left_ratio = abs(left_ear.x - nose_x)
    right_ratio = abs(right_ear.x - nose_x)
    # Using angle velocity
    angular_vel, yaw_angle = angular_velocity_calc(landmarks)
    if previous_ratio is None:
        previous_ratio = left_ratio - right_ratio
    if previous_yaw is None:
        previous_yaw = yaw_angle
    if angular_vel > angular_velocity_threshold:  # Full turns are much slower
        if (left_ratio - right_ratio) > previous_ratio + shake_threshold and head_state != "left" and abs(yaw_angle - previous_yaw) < 25:
            head_state = "left"
            shake_counter += 1
            detected_shake = True
            print(f'Shake {shake_counter} detected! Speed: {angular_vel} Yaw angle: {yaw_angle}')
        elif (left_ratio - right_ratio) < previous_ratio - shake_threshold and head_state != "right" and abs(yaw_angle - previous_yaw) < 25:
            head_state = "right"
            shake_counter += 1
            detected_shake = True
            print(f'Shake {shake_counter} detected! Speed: {angular_vel} Yaw angle: {yaw_angle}')
    if detected_shake:
        last_shake_time = time.time()
    if shake_counter == 1 and start_time is None:
        start_time = time.time()
    if last_shake_time is not None:
        elapsed_time = time.time() - last_shake_time
        if elapsed_time > 3 and shake_counter<3:
            start_time = None
            shake_counter = 0
            last_shake_time = None
            print('Headshake deactivation failed: Time taken too long')
    if shake_counter == shake_number:
        trigger = 0
        shake_counter = 0
        last_shake_time = None
        print('Deactivation Successfull!')
    previous_ratio = left_ratio - right_ratio
    previous_yaw = yaw_angle
    return trigger, angular_vel
def head_width(landmarks):
    left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
    right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value]
    return abs(right_ear.x - left_ear.x)

REFERENCE_S_DISTANCE = 0.345
REFERENCE_DISTANCE = 1.1
ref_nose_z = 0.8
ref_nose = 1.2
ref_s_z = 0.8
ref_s = 1.2
def distance_camera_human(landmarks):
    global ref_distance_train, ref_angle_train, ref_shoulder_z
    distance_shoulder = shoulder_width(landmarks)
    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    #dist_cam1 = np.sqrt((REFERENCE_DISTANCE * (REFERENCE_YAW_DISTANCE/distance_shoulder))**2+(ref_nose * (ref_nose_z/nose.z))**2)
    dist_cam1 = (1/((nose.z)/1))*ref_nose*ref_nose_z
    return dist_cam1
def yaw_correction(yaw_angle):
        """DAMPING EQUATION not in use ="""
        a, b = 0.005, 0.00009  # Tuned coefficients
        return 1 + a * abs(yaw_angle) + b * (yaw_angle ** 2)
def distance_camera_human2_0(landmarks):
    global REFERENCE_S_DISTANCE, REFERENCE_DISTANCE
    should_width = shoulder_width(landmarks)
    yaw_angle = calculate_yaw_angle(landmarks)
    yaw_angle_rad_cos = float(np.cos(np.radians(yaw_angle)))
    if should_width == 0:
        return None
    if yaw_angle == 0:
        return None
    D = REFERENCE_DISTANCE * (REFERENCE_S_DISTANCE / (should_width))* yaw_angle_rad_cos
    return D

def feature_names():
    body_part_iteration = 1
    body_part_name = []

    for landmark_point in mp_pose.PoseLandmark:
        lm_value = landmark_point.value
        lm_name = landmark_point.name
        if body_part_iteration < 2:
            body_part_name.append(lm_name + "_x")
            body_part_name.append(lm_name + "_y")
            body_part_name.append(lm_name + "_z")
    body_part_iteration = body_part_iteration + 1
    body_part_name = np.append(body_part_name, (
    "Torso Angle in Degrees", "Center of Mass", "Knee-Hip Angle in Degrees", "Knee-Hip Displacement"))
    return body_part_name

# ======= Fall detection threshold addition =====
vel_hip_array = []
vel_hip_y = 0
def velocity_hip_y(landmarks):
    global vel_hip_array, vel_hip_y

    # Yaw angle
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    mid_value_y = (left_hip.y+right_hip.y)/2
    vel_hip_array.append(mid_value_y)
    if len(vel_hip_array) > 5:  # Keep the last 5 yaw values
        vel_hip_array.pop(0)
    if len(vel_hip_array) >= 2:
        # To get the angular velocity in degrees per second multiply the equation below by the frame rate of the camera. Right now it is degrees per frame
        vel_hip_y = abs(vel_hip_array[-1] - vel_hip_array[0]) / len(vel_hip_array)
    return vel_hip_y, mid_value_y
prev_velocity = None
speed_condition = 0
def fall_conditions(landmarks):
    global prev_velocity, speed_condition
    vel_hip_y_ = velocity_hip_y(landmarks)
    threshold = prev_velocity*1.5
    if prev_velocity is None:
        prev_velocity = vel_hip_y_
    if vel_hip_y_ > threshold:
        speed_condition = 1
    else:
        speed_condition = 0
    return speed_condition, vel_hip_y_


def plot_roc_curve(models, X_test, y_test, title):
    """
    Plot ROC curves for multiple models.

    :param models: Dictionary of trained models (e.g., {"KNN": knn_model, "SVM": svm_model})
    :param X_test: Test dataset features
    :param y_test: Test dataset labels
    """

    plt.figure(figsize=(11, 8))

    for name, model in models.items():
        # Get model predictions (probabilities)
        if hasattr(model, "predict_proba"):  # For models with probability estimates
            y_scores = model.predict_proba(X_test)[:, 1]  # Take probability of class 1
        else:  # For models without predict_proba (like SVM with linear kernel)
            y_scores = model.decision_function(X_test)

        # Compute ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_test, y_scores)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})", linewidth=4)

    # Plot reference line (random guessing)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guessing", linewidth=4)

    # Formatting the plot
    plt.xlabel("False Positive Rate (FPR)", fontsize =18)
    plt.ylabel("True Positive Rate (TPR)", fontsize = 18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title(title, fontsize = 22)
    plt.legend(loc="lower right", fontsize=16)
    plt.grid()
    plt.savefig(title)
    plt.show()
