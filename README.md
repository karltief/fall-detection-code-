# RoboMaster Tello TT – Fall & Emergency-Gesture Detection  
_BEng EEE Honours Project 2024-25_

> Turn a DJI RoboMaster **Tello Talent** drone into a fall-aware flying assistant.

---

## Contents

| File / Dir | Purpose |
|------------|---------|
| **main.py** | Primary entry-point – connects to drone/camera, loads chosen model, runs detection loop. |
| **main_m1.py** | Alternate entry-point for “method 1” experiments (kept for reproducibility). |
| **camera.py** | Generic USB/web-cam frame capture. |
| **camera_drone.py** | Tello video-stream wrapper built on `djitellopy`. |
| **functions.py** | Utility helpers (frame pre-proc, drawing bboxes, SMS/e-mail alerts, logging). |
| **KNN_meth1.py** / **KNN_meth2.py** | K-Nearest-Neighbours fall-detection pipelines (two variants). |
| **LogReg_m1.py** / **LogReg_m2.py** | Logistic-regression versions. |
| **SVM_meth1.py** / **SVM_meth2.py** | Support-Vector-Machine versions. |
| **C_parameter_*.py** | Grid-search scripts to tune the `C` hyper-parameter for LR & SVM (baseline vs. method 1). |
| **README.md** | This file. |

> **\_m1”**  
> “method 1” refers to dataset 1
> \download the dataset from https://www.kaggle.com/datasets/uttejkumarkandagatla/fall-detection-dataset
> **\_m2”**  
> “method 2” refers to dataset 2
> \download the dataset from https://www.kaggle.com/datasets/tuyenldvn/caucafall
---



