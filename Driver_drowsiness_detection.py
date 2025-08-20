import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils
import time
import pygame
from pygame import mixer
import os

# Initialize pygame mixer
pygame.mixer.init()
try:
    alarm_sound = mixer.Sound("alarm.wav")
except:
    print("Warning: Could not load alarm sound file. Using beep instead.")
    alarm_sound = None

# Constants
EYE_AR_THRESH = 0.25  # Default threshold (will be calibrated)
EYE_AR_CONSEC_FRAMES = 15
YAWN_THRESH = 20
HEAD_TILT_THRESH = 15
ALARM_RESET_TIME = 3  # seconds

# Global variables
COUNTER = 0
ALARM_ON = False
SYSTEM_ACTIVE = True
CALIBRATED = False

# Initialize dlib's face detector and facial landmark predictor
try:
    detector = dlib.get_frontal_face_detector()
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    if not os.path.exists(predictor_path):
        raise FileNotFoundError("Landmark predictor file not found")
    predictor = dlib.shape_predictor(predictor_path)
except Exception as e:
    print(f"Error initializing face detector: {e}")
    exit()

# Facial landmarks indexes
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]


def eye_aspect_ratio(eye):
    try:
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear
    except:
        return None


def mouth_aspect_ratio(mouth):
    try:
        A = dist.euclidean(mouth[13], mouth[19])
        B = dist.euclidean(mouth[14], mouth[18])
        C = dist.euclidean(mouth[15], mouth[17])
        D = dist.euclidean(mouth[12], mouth[16])
        mar = (A + B + C) / (3.0 * D)
        return mar
    except:
        return None


def head_pose_estimation(shape, size):
    try:
        image_points = np.array(
            [shape[30], shape[8], shape[36], shape[45], shape[48], shape[54]],
            dtype="double",
        )

        focal_length = size[1]
        center = (size[1] / 2, size[0] / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
            dtype="double",
        )

        dist_coeffs = np.zeros((4, 1))
        (_, rotation_vector, _) = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs
        )

        rmat, _ = cv2.Rodrigues(rotation_vector)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        return angles
    except:
        return None


def calibrate_thresholds(cap, num_frames=30):
    global EYE_AR_THRESH, CALIBRATED
    ear_values = []

    print("Calibrating - please keep your eyes open normally...")
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            if leftEAR and rightEAR:
                ear = (leftEAR + rightEAR) / 2.0
                ear_values.append(ear)

        cv2.putText(
            frame,
            "Calibrating...",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Calibration", frame)
        cv2.waitKey(1)

    if ear_values:
        avg_ear = sum(ear_values) / len(ear_values)
        EYE_AR_THRESH = avg_ear * 0.75  # Set threshold to 75% of open eye value
        CALIBRATED = True
        print(f"Calibration complete. New EAR threshold: {EYE_AR_THRESH:.2f}")
    else:
        print("Calibration failed - using default threshold")

    cv2.destroyWindow("Calibration")


def play_alarm():
    global ALARM_ON
    if alarm_sound:
        alarm_sound.play(-1)  # Loop the alarm
    else:
        # Fallback beep
        for _ in range(3):
            print("\a", end="", flush=True)
            time.sleep(0.5)
    ALARM_ON = True


def stop_alarm():
    global ALARM_ON
    if alarm_sound:
        alarm_sound.stop()
    ALARM_ON = False


def detection_loop():
    global COUNTER, ALARM_ON, SYSTEM_ACTIVE, CALIBRATED

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture")
        return

    # Calibrate thresholds
    calibrate_thresholds(cap)

    while SYSTEM_ACTIVE:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        # Reset detection flags
        ear = None
        mar = None
        angles = None

        for rect in rects:
            try:
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                # Eye detection
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)

                if leftEAR and rightEAR:
                    ear = (leftEAR + rightEAR) / 2.0

                    # Eye closure detection
                    if ear < EYE_AR_THRESH:
                        COUNTER += 1
                        if COUNTER >= EYE_AR_CONSEC_FRAMES and not ALARM_ON:
                            play_alarm()
                    else:
                        COUNTER = 0
                        if ALARM_ON:
                            stop_alarm()

                # Mouth detection
                mouth = shape[mStart:mEnd]
                mar = mouth_aspect_ratio(mouth)
                if mar and mar > YAWN_THRESH and not ALARM_ON:
                    play_alarm()

                # Head pose estimation
                angles = head_pose_estimation(shape, frame.shape)
                if angles and (
                    abs(angles[0]) > HEAD_TILT_THRESH
                    or abs(angles[1]) > HEAD_TILT_THRESH
                ):
                    if not ALARM_ON:
                        play_alarm()

                # Draw landmarks
                if ear:
                    leftEyeHull = cv2.convexHull(leftEye)
                    rightEyeHull = cv2.convexHull(rightEye)
                    cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                    cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

                if mar:
                    mouthHull = cv2.convexHull(mouth)
                    cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

            except Exception as e:
                print(f"Detection error: {e}")
                continue

        # Display information
        status_color = (0, 0, 255) if ALARM_ON else (0, 255, 0)
        cv2.putText(
            frame,
            f"Alarm: {'ON' if ALARM_ON else 'OFF'}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            status_color,
            2,
        )

        if ear is not None:
            cv2.putText(
                frame,
                f"EAR: {ear:.2f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

        if mar is not None:
            cv2.putText(
                frame,
                f"MAR: {mar:.2f}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

        if angles is not None:
            cv2.putText(
                frame,
                f"Head Pose: {angles[0]:.1f}, {angles[1]:.1f}",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

        if not CALIBRATED:
            cv2.putText(
                frame,
                "NOT CALIBRATED",
                (frame.shape[1] - 200, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        cv2.imshow("Driver Drowsiness Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            SYSTEM_ACTIVE = False
        elif key == ord("r"):
            stop_alarm()
            COUNTER = 0

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("Starting Driver Drowsiness Detection System...")
    detection_loop()
    if ALARM_ON:
        stop_alarm()
    pygame.quit()
    print("System shutdown complete")
