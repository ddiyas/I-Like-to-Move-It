import cv2
import mediapipe as mp
from collections import deque

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)

position_history = deque(maxlen=10)

jump_threshold = 0.04
is_jumping = False
jump_cooldown = 0

# Left/Right movement detection
left_right_threshold = 0.06
is_moving_left = False
is_moving_right = False
left_cooldown = 0
right_cooldown = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    marker_frame = frame.copy()
    marker_frame[:] = 0
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        # Draw body landmarks (excluding face)
        # We'll manually draw connections and skip face landmarks
        landmarks = results.pose_landmarks.landmark
        h, w, _ = marker_frame.shape

        # Draw all body connections except face
        for connection in mp_pose.POSE_CONNECTIONS:
            start_idx = connection[0]
            end_idx = connection[1]

            # Skip face landmarks (0-10 are face/head area)
            if start_idx <= 10 or end_idx <= 10:
                continue

            start = landmarks[start_idx]
            end = landmarks[end_idx]

            start_point = (int(start.x * w), int(start.y * h))
            end_point = (int(end.x * w), int(end.y * h))

            cv2.line(marker_frame, start_point, end_point, (0, 255, 0), 2)

        # Draw body keypoints (excluding face)
        for idx, landmark in enumerate(landmarks):
            if idx <= 10:  # Skip face landmarks
                continue
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(marker_frame, (x, y), 5, (0, 0, 255), -1)

        # Draw a circle for the head/face
        nose = landmarks[mp_pose.PoseLandmark.NOSE]
        nose_x = int(nose.x * w)
        nose_y = int(nose.y * h)

        # Calculate head radius based on shoulder width
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        shoulder_width = abs(left_shoulder.x - right_shoulder.x) * w
        head_radius = int(shoulder_width * 0.3)  # Head is ~30% of shoulder width

        cv2.circle(marker_frame, (nose_x, nose_y), head_radius, (255, 255, 0), 2)

        left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
        left_shoulder = results.pose_landmarks.landmark[
            mp_pose.PoseLandmark.LEFT_SHOULDER
        ]
        right_shoulder = results.pose_landmarks.landmark[
            mp_pose.PoseLandmark.RIGHT_SHOULDER
        ]
        hip_y = (left_hip.y + right_hip.y) / 2
        hip_x = (left_hip.x + right_hip.x) / 2
        shoulder_y = (left_shoulder.y + right_shoulder.y) / 2

        position_history.append((hip_y, hip_x, shoulder_y))

        if len(position_history) >= 5:
            past_hip_y, past_hip_x, past_shoulder_y = position_history[0]
            hip_movement = past_hip_y - hip_y
            shoulder_movement = past_shoulder_y - shoulder_y
            vertical_movement = (hip_movement + shoulder_movement) / 2

            # Horizontal movement (positive = moving right, negative = moving left)
            horizontal_movement = hip_x - past_hip_x

            # Jump detection
            if (
                vertical_movement > jump_threshold
                and not is_jumping
                and jump_cooldown == 0
            ):
                print("JUMP DETECTED!")
                is_jumping = True
                jump_cooldown = 20

                cv2.putText(
                    marker_frame,
                    "JUMP!",
                    (200, 200),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    3,
                    (0, 255, 0),
                    5,
                )

            if abs(vertical_movement) < 0.02:
                is_jumping = False

            if jump_cooldown > 0:
                jump_cooldown -= 1

            # Left movement detection
            if (
                horizontal_movement < -left_right_threshold
                and not is_moving_left
                and left_cooldown == 0
            ):
                print("LEFT MOVEMENT DETECTED!")
                is_moving_left = True
                left_cooldown = 20

                cv2.putText(
                    marker_frame,
                    "LEFT!",
                    (50, 200),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    3,
                    (255, 0, 255),
                    5,
                )

            if abs(horizontal_movement) < 0.02:
                is_moving_left = False

            if left_cooldown > 0:
                left_cooldown -= 1

            # Right movement detection
            if (
                horizontal_movement > left_right_threshold
                and not is_moving_right
                and right_cooldown == 0
            ):
                print("RIGHT MOVEMENT DETECTED!")
                is_moving_right = True
                right_cooldown = 20

                cv2.putText(
                    marker_frame,
                    "RIGHT!",
                    (350, 200),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    3,
                    (0, 255, 255),
                    5,
                )

            if abs(horizontal_movement) < 0.02:
                is_moving_right = False

            if right_cooldown > 0:
                right_cooldown -= 1

            cv2.putText(
                marker_frame,
                f"Vertical: {vertical_movement:.3f} Horizontal: {horizontal_movement:.3f}",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
        else:
            cv2.putText(
                marker_frame,
                f"Initializing... {len(position_history)}/5",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2,
            )

    cv2.imshow("Jump Detection", marker_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
