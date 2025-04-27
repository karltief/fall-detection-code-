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

> **Why so many “\_m1” files?**  
> “method 1” is the lightweight pipeline optimised for real-time onboard inference.  
> Files without `_m1` are the original, higher-accuracy but slower baselines.

---

## Quick-Start

### 1  Set-up

```bash
# Clone & enter repo
git clone https://github.com/<your-user>/<repo>.git
cd <repo>

# Python 3.10+ venv
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate

# Install deps
pip install -r requirements.txt   # OpenCV, scikit-learn, djitellopy, etc.

