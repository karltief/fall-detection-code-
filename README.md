# fall-detection-code-
 
_BEng Electronics & Electrical Engineering Honours Project (2024-25)_

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](#) 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) 
![CI](https://img.shields.io/github/actions/workflow/status/<your-user>/<repo>/ci.yml)

A cross-platform (Linux & Windows) Python project that turns a **DJI RoboMaster Tello Talent** drone into an autonomous guardian:

* **Fall Detection** – real-time classification of human falls from the onboard camera using traditional ML (K-NN, SVM) and CNN-based models.  
* **Emergency Gestures** – recognizes a _“Help!”_ gesture to trigger assistance and a _“Safe”_ gesture to deactivate alerts.  
* **Caregiver Alerts** – sends configurable SMS/e-mail notifications with last-seen image & GPS coordinates.  
* **Safe Flight Control** – handles take-off, hover, patrol, and landing through `djitellopy`, with geofence & battery safeguards.  

---

## 1. Project Structure

