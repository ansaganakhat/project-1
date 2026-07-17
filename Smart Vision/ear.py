import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import distance as dist
import platform
import os
import time

# ============================================
# SAFE BEEP FUNCTION
# ============================================
def beep():
    try:
        if platform.system() == "Windows":
            import winsound
            winsound.Beep(2000, 300)
        else:
            os.system('printf "\a"')
    except:
        pass


# ============================================
# MEDIAPIPE SETUP
# ============================================
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.7
)

# ============================================
# FACIAL LANDMARKS
# ============================================
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

MOUTH = [61, 291, 39, 181, 0, 17, 405, 314]

NOSE = 1
CHIN = 152


# ============================================
# EYE ASPECT RATIO (EAR)
# ============================================
def eye_aspect_ratio(eye):
    """
    EAR = eye openness measurement
    """

    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])

    if C == 0:
        return 0.0

    return (A + B) / (2.0 * C)


# ============================================
# MOUTH ASPECT RATIO (MAR)
# ============================================
def mouth_aspect_ratio(mouth):
    """
    MAR = yawn detection
    """

    A = dist.euclidean(mouth[2], mouth[6])
    B = dist.euclidean(mouth[3], mouth[7])
    C = dist.euclidean(mouth[0], mouth[1])

    if C == 0:
        return 0.0

    return (A + B) / (2.0 * C)


# ============================================
# FATIGUE DETECTOR CLASS
# ============================================
class FatigueDetector:

    def __init__(self):

        self.score = 0

        # eye closed frame counter
        self.closed_eye_frames = 0

        # alert threshold
        self.ALERT_THRESHOLD = 45

        # beep cooldown
        self.last_beep_time = 0

    # ========================================
    # HEAD TILT
    # ========================================
    def head_tilt(self, landmarks, h):

        try:
            nose = landmarks.landmark[NOSE]
            chin = landmarks.landmark[CHIN]

            return (chin.y - nose.y) * h

        except:
            return 0

    # ========================================
    # FATIGUE SCORE
    # ========================================
    def update_score(self, ear, mar, head_tilt):

        score = 0

        # ------------------------------------
        # EYES CLOSED
        # ------------------------------------
        if ear < 0.22:
            score += 25

        # ------------------------------------
        # YAWNING
        # ------------------------------------
        if mar > 0.60:
            score += 20

        # ------------------------------------
        # HEAD DOWN
        # ------------------------------------
        if head_tilt > 120:
            score += 25

        # ------------------------------------
        # LONG EYE CLOSURE
        # ------------------------------------
        if self.closed_eye_frames > 15:
            score += 20

        # ------------------------------------
        # SMOOTHING
        # ------------------------------------
        self.score = int(self.score * 0.5 + score * 0.5)

        return self.score

    # ========================================
    # ALERT SYSTEM
    # ========================================
    def alert(self, frame):

        if self.score >= self.ALERT_THRESHOLD:

            cv2.putText(
                frame,
                "DROWSINESS ALERT!",
                (50, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),
                3
            )

            current_time = time.time()

            # beep every 2 seconds only
            if current_time - self.last_beep_time > 2:
                beep()
                self.last_beep_time = current_time


# ============================================
# CAMERA INITIALIZATION
# ============================================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Camera not found!")
    exit()

detector = FatigueDetector()

print("===================================")
print("Stable Driver Drowsiness System")
print("Press ESC to Exit")
print("===================================")

# ============================================
# MAIN LOOP
# ============================================
while True:

    ret, frame = cap.read()

    if not ret or frame is None:
        continue

    # mirror effect
    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape

    # BGR -> RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ========================================
    # FACE DETECTION
    # ========================================
    try:
        results = face_mesh.process(rgb)
    except:
        continue

    # ========================================
    # IF FACE FOUND
    # ========================================
    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            # ====================================
            # LEFT EYE
            # ====================================
            left_eye = []

            for idx in LEFT_EYE:
                point = face_landmarks.landmark[idx]

                left_eye.append([
                    point.x * w,
                    point.y * h
                ])

            # ====================================
            # RIGHT EYE
            # ====================================
            right_eye = []

            for idx in RIGHT_EYE:
                point = face_landmarks.landmark[idx]

                right_eye.append([
                    point.x * w,
                    point.y * h
                ])

            # ====================================
            # EAR
            # ====================================
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)

            ear = (left_ear + right_ear) / 2.0

            # ====================================
            # MOUTH
            # ====================================
            mouth = []

            for idx in MOUTH:
                point = face_landmarks.landmark[idx]

                mouth.append([
                    point.x * w,
                    point.y * h
                ])

            # ====================================
            # MAR
            # ====================================
            mar = mouth_aspect_ratio(mouth)

            # ====================================
            # HEAD TILT
            # ====================================
            head_tilt = detector.head_tilt(face_landmarks, h)

            # ====================================
            # EYE CLOSED COUNTER
            # ====================================
            if ear < 0.22:
                detector.closed_eye_frames += 1
            else:
                detector.closed_eye_frames = 0

            # ====================================
            # SCORE UPDATE
            # ====================================
            score = detector.update_score(
                ear,
                mar,
                head_tilt
            )

            # ====================================
            # UI TEXT
            # ====================================
            cv2.putText(
                frame,
                f"EAR: {ear:.2f}",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"MAR: {mar:.2f}",
                (30, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"SCORE: {score}",
                (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"EYE CLOSED FRAMES: {detector.closed_eye_frames}",
                (30, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (200, 200, 200),
                2
            )

            # ====================================
            # ALERT
            # ====================================
            detector.alert(frame)

    else:
        cv2.putText(
            frame,
            "NO FACE DETECTED",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    # ========================================
    # SHOW WINDOW
    # ========================================
    cv2.imshow(
        "STABLE DRIVER DROWSINESS SYSTEM",
        frame
    )

    # ESC KEY
    if cv2.waitKey(1) & 0xFF == 27:
        break

# ============================================
# RELEASE
# ============================================
cap.release()
cv2.destroyAllWindows()
