from functions import *
global body_part_name
from sklearn.model_selection import train_test_split

df = pd.read_csv("output_m2.csv")


# Label the columns for the classification process
X = df[df.columns[0:103]]
y = df[["Fall/NoFALL(1,0)"]].values.ravel()
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
# Choose a dataset
dataset=1
if dataset == 1:
    knn = joblib.load("best_knn_model.pkl")
    svm = joblib.load("svm_model.pkl")
    LogReg = joblib.load("Method_1/LogReg_model.pkl")
elif dataset == 2:
    knn = joblib.load("best_knn_model_m2.pkl")
    svm = joblib.load("svm_m2_model.pkl")
    LogReg = joblib.load("Method_2/LogReg_model.pkl")
models = {
    "KNN": knn,  # Pre-trained KNN model
    "SVM": svm,  # Pre-trained SVM model
    "LogReg": LogReg  # Pre-trained Logistic Regression model
}

# Plot the ROC Curve comparison between the models for the chosen dataset
plot_roc_curve(models, X_test, y_test, title=f"ROC Curve Comparison for Dataset {dataset}")

model = "SVM_m2_function_thresh_fall"

# Initialise the pose estimation model
pose, mp_drawing, mp_drawing_styles, mp_pose = initialisation_mp()


snapshot = False
cap = cv2.VideoCapture(0)  # Open webcam
while True:
    ret, frame = cap.read()
    if not ret:
        print("No frame received. Exiting...")
        break

    # Convert BGR to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    #face_results = face_mesh.process(frame_rgb)

    if results.pose_landmarks:
        # Draw pose landmarks
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )

        # Extract features from pose landmarks
        landmarks = results.pose_landmarks.landmark
        # Open arrays for storing the real-time values
        coords = []
        coords_fall = []
        pose_array_fall = []
        # Frame shape
        frame_height, frame_width, _ = frame.shape
        for landmark_point in mp_pose.PoseLandmark:
            lm_value = landmark_point.value
            lm_name = landmark_point.name
            coords.append(landmarks[landmark_point].x)
            coords.append(landmarks[landmark_point].y)
            coords.append(landmarks[landmark_point].z)

        # Torso Angle, COM, calculation
        torso_angle = calculate_torso_angle(landmarks).ravel()
        com = calculate_com_displacement(landmarks).ravel()
        knee_hip_angle = calculate_knee_hip_angle(landmarks).ravel()
        knee_hip_displacement = calculate_knee_hip_displacement(landmarks).ravel()

        x1, y1, x2, y2 = get_bounding_box(landmarks, frame_width, frame_height)

        coords_fall = np.append(coords, [torso_angle, com, knee_hip_angle, knee_hip_displacement])
        pose_array_fall = np.append(pose_array_fall, [coords_fall]).reshape(-1,103)
        feature_name = feature_names()
        x_live = pd.DataFrame(pose_array_fall, columns = feature_name)

        # Using the live pose landmarks - predict the class (fall/no-fall)
        prediction_knn = knn.predict(x_live)[0]
        prediction_svm = svm.predict(x_live)[0]
        prediction_LogReg = LogReg.predict(x_live)[0]
        # Predict the emergency gesture trigger using teh coordinates and the threshold-based algorithm
        emergency_prediction, dist_shoulder = emergency_gesture_rec(coords, landmarks)
        ang_vel, yaw_angle = angular_velocity_calc(landmarks)
        dist_cam = distance_camera_human2_0(landmarks)
        _, vel_hip_y = velocity_hip_y(landmarks)

        # Display the distance, yaw angle and angular velocity live
        cv2.putText(frame, f"Ang Vel: {ang_vel:.2f} deg/frame", (20, 430), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Yaw Ang: {yaw_angle:.2f} deg", (20, 460), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Distance: {dist_cam:.2f}m", (20, 400), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        # Display a trigger alert for fall detection or gesture recognition if the prediction is high
        if emergency_prediction == 1:
            cv2.putText(frame, "Alert!", (20, 70), cv2.FONT_HERSHEY_TRIPLEX, 0.9, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if prediction_LogReg == 1:
            cv2.putText(frame, "FALL DETECTED!", (20, 30), cv2.FONT_HERSHEY_TRIPLEX,0.9, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        else:
            cv2.putText(frame, "No Fall", (20, 30), cv2.FONT_HERSHEY_TRIPLEX,0.9, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Show the video feed
    frame = cv2.resize(frame, None, fx=1.5, fy=1.5)
    cv2.imshow("Fall Detection", frame)

    # Exit on 'q' or ESC key
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        print("Snapshot in 5 Seconds")
        snapshot_time = time.time()+7
        snapshot = True
    # If 's' key is pressed, wait five seconds and take a screenshot
    if snapshot and time.time() >= snapshot_time:
        snapshot_filename = (f"snapshots/{time.time()}.png")
        cv2.imwrite(snapshot_filename, frame)
        print(f"Snapshot saved as {snapshot_filename}")
        snapshot = False
    if key == 27 or key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()




