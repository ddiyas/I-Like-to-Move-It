import cv2
import mediapipe as mp

try:
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
except Exception as e:
    print(f"ERROR during initialization: {e}")
    import traceback

    traceback.print_exc()
    exit()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Cannot access webcam!")
    exit()

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("ERROR: Can't read frame from webcam")
        break

    frame_count += 1
    if frame_count % 30 == 0:
        print(f"Processing frame {frame_count}...")

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )

    cv2.imshow("CV Controller", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Quitting")
