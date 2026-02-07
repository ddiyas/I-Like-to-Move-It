import cv2
import mediapipe as mp
from collections import deque

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)

position_history = deque(maxlen=10)

jump_threshold = 0.08
is_jumping = False
cooldown = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )

        left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
        left_shoulder = results.pose_landmarks.landmark[
            mp_pose.PoseLandmark.LEFT_SHOULDER
        ]
        right_shoulder = results.pose_landmarks.landmark[
            mp_pose.PoseLandmark.RIGHT_SHOULDER
        ]
        hip_y = (left_hip.y + right_hip.y) / 2
        shoulder_y = (left_shoulder.y + right_shoulder.y) / 2

        position_history.append((hip_y, shoulder_y))

        if len(position_history) >= 5:
            past_hip_y, past_shoulder_y = position_history[0]
            hip_movement = past_hip_y - hip_y
            shoulder_movement = past_shoulder_y - shoulder_y
            movement = (hip_movement + shoulder_movement) / 2

            if movement > jump_threshold and not is_jumping and cooldown == 0:
                print("🏃 JUMP DETECTED!")
                is_jumping = True
                cooldown = 20

                cv2.putText(
                    frame,
                    "JUMP!",
                    (200, 200),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    3,
                    (0, 255, 0),
                    5,
                )

            if abs(movement) < 0.02:
                is_jumping = False

            if cooldown > 0:
                cooldown -= 1

            cv2.putText(
                frame,
                f"Movement: {movement:.3f}",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
        else:
            cv2.putText(
                frame,
                f"Initializing... {len(position_history)}/5",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2,
            )

    cv2.imshow("Jump Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
