
# Driver Drowsiness Detection System 🚗💤

This project is a **real-time driver drowsiness detection system** built using **OpenCV, dlib, NumPy, and Pygame**.
It detects **eye closure, yawning, and head tilt** to determine signs of drowsiness and alerts the driver with an alarm.

---

## 🔹 Features

* **Eye Aspect Ratio (EAR):** Detects prolonged eye closure.
* **Mouth Aspect Ratio (MAR):** Detects yawning.
* **Head Pose Estimation:** Detects unusual head tilts.
* **Calibration:** Adjusts eye aspect ratio threshold automatically for individual users.
* **Alarm System:** Plays an alarm sound when drowsiness is detected.
* **Visualization:** Displays EAR, MAR, and head pose on the video feed.

---

## 🔹 Requirements

* Python 3.7+
* Install the required dependencies:

```bash
pip install opencv-python dlib imutils numpy scipy pygame
```

You will also need the **dlib facial landmark predictor file**:
👉 [Download shape\_predictor\_68\_face\_landmarks.dat](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
Extract and place it in the project directory.

---

## 🔹 Files

* `main.py` → Driver Drowsiness Detection System
* `alarm.wav` → Alarm sound file (place in same directory)
* `shape_predictor_68_face_landmarks.dat` → Pre-trained facial landmark predictor

---

## 🔹 Usage

1. Clone the repository or download the project files.
2. Place the `shape_predictor_68_face_landmarks.dat` and `alarm.wav` files in the same folder.
3. Run the script:

```bash
python main.py
```

4. The system will first **calibrate** your eye aspect ratio.

   * Keep your eyes open normally during calibration.
5. The system starts detecting drowsiness in real time.

### Controls

* Press **`q`** → Quit the program.
* Press **`r`** → Reset alarm & counter.

---

## 🔹 How it Works

1. **Facial landmarks** are detected using **dlib**.
2. **EAR (Eye Aspect Ratio)** is calculated to detect closed eyes.
3. **MAR (Mouth Aspect Ratio)** is calculated to detect yawns.
4. **Head Pose Estimation** is done using OpenCV’s `solvePnP`.
5. If drowsiness is detected, an **alarm** is triggered.

---

## 🔹 Demo Output

* **Green Contours** → Eyes and Mouth detected.
* **Text Overlay** → EAR, MAR, and Head Pose angles displayed.
* **Alarm ON/OFF** status shown on screen.

---

## 🔹 Future Improvements

* Add blink count tracking.
* Integrate with vehicle IoT systems.
* Use deep learning models for more accuracy.
* Send alerts to connected devices (mobile notifications).

---

## 🔹 Disclaimer

⚠️ This project is for **educational and research purposes only**.
Do not rely solely on this system for driver safety.
