import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def compute_q_angle(hip, knee, ankle):
    hip   = np.array(hip,   dtype=float)
    knee  = np.array(knee,  dtype=float)
    ankle = np.array(ankle, dtype=float)

    v1 = hip   - knee
    v2 = ankle - knee

    cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    cosine = np.clip(cosine, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))

def extend_line(p1, p2, length=80):
    p1, p2 = np.array(p1), np.array(p2)
    direction = p2 - p1
    direction = direction / (np.linalg.norm(direction) + 1e-6)
    return (p2 + direction * length).astype(int)


cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        h, w = frame.shape[:2]

        hip   = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
        knee  = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
        ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]

        hip_pt   = [hip.x   * w, hip.y * h]
        knee_pt  = [knee.x  * w, knee.y * h]
        ankle_pt = [ankle.x * w, ankle.y * h]

        angle = compute_q_angle(hip_pt, knee_pt, ankle_pt)

        hip_extended   = extend_line(hip_pt,   knee_pt)
        ankle_extended = extend_line(ankle_pt, knee_pt)

        # Line 1: Hip → Knee → extended (blue)
        cv2.line(frame,
                 (int(hip_pt[0]), int(hip_pt[1])),
                 tuple(hip_extended), (255, 80, 0), 3)

        # Line 2: Ankle → Knee → extended (yellow)
        cv2.line(frame,
                 (int(ankle_pt[0]), int(ankle_pt[1])),
                 tuple(ankle_extended), (0, 220, 255), 3)

        # Joints
        cv2.circle(frame, (int(hip_pt[0]),   int(hip_pt[1])),   7, (255,  80,   0), -1)
        cv2.circle(frame, (int(knee_pt[0]),  int(knee_pt[1])),  8, (  0, 255,   0), -1)
        cv2.circle(frame, (int(ankle_pt[0]), int(ankle_pt[1])), 7, (  0, 220, 255), -1)

        # Q-angle only
        cv2.putText(frame,
                    f"Q-angle: {angle:.1f} deg",
                    (30, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)

    cv2.imshow("Real-Time Q-Angle", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
