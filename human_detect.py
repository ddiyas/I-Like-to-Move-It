import cv2
import mediapipe as mp
from collections import deque

try:
    from CalorieTracker import CalorieTracker

    HAS_CALORIE_TRACKER = True
except ImportError:
    print("⚠️ CalorieTracker not found - running without calorie tracking")
    HAS_CALORIE_TRACKER = False

    class CalorieTracker:
        def __init__(self, weight):
            pass

        def start_session(self):
            pass

        def update(self, jumping, moving):
            pass

        def get_calories(self):
            return 0

        def get_session_time(self):
            return 0.0

        def end_session(self):
            pass


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

print("📷 Opening camera...")
try:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Cannot open camera! Make sure your webcam is connected.")
    print("✅ Camera opened successfully!")
except Exception as e:
    print(f"❌ Camera Error: {e}")
    exit(1)

weight_kg = 130
calorie_tracker = CalorieTracker(weight_kg)
calorie_tracker.start_session()

position_history = deque(maxlen=10)
arm_history = deque(maxlen=10)  # Track arm positions for jogging detection


class PlayerState:
    def __init__(self):
        self.is_jumping = False
        self.is_ducking = False
        self.jump_timer = 0
        self.duck_timer = 0

        self.is_moving_left = False
        self.is_moving_right = False
        self.left_timer = 0
        self.right_timer = 0

        self.is_jogging = False
        self.jog_timer = 0

        self.left_cooldown = 0
        self.right_cooldown = 0
        self.jump_cooldown = 0
        self.duck_cooldown = 0

    def update_timers(self):
        if self.jump_timer > 0:
            self.jump_timer -= 1
        if self.duck_timer > 0:
            self.duck_timer -= 1
        if self.left_timer > 0:
            self.left_timer -= 1
        if self.right_timer > 0:
            self.right_timer -= 1
        if self.jog_timer > 0:
            self.jog_timer -= 1
        if self.left_cooldown > 0:
            self.left_cooldown -= 1
        if self.right_cooldown > 0:
            self.right_cooldown -= 1
        if self.jump_cooldown > 0:
            self.jump_cooldown -= 1
        if self.duck_cooldown > 0:
            self.duck_cooldown -= 1

        if self.jump_timer == 0:
            self.is_jumping = False
        if self.duck_timer == 0:
            self.is_ducking = False
        if self.left_timer == 0:
            self.is_moving_left = False
        if self.right_timer == 0:
            self.is_moving_right = False
        if self.jog_timer == 0:
            self.is_jogging = False


player = PlayerState()

VERTICAL_THRESHOLD = 0.05
HORIZONTAL_THRESHOLD = 0.08
ARM_MOVEMENT_THRESHOLD = 0.04  # Threshold for detecting arm pumping
JOG_DETECTION_FRAMES = 12  # Number of frames to analyze for jogging
JUMP_DURATION = 15
DUCK_DURATION = 20
JOG_DURATION = 25
COOLDOWN_DURATION = 10


def get_body_center_y(landmarks):
    """Calculate the center Y position of the torso"""
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

    avg_y = (left_shoulder.y + right_shoulder.y + left_hip.y + right_hip.y) / 4
    return avg_y


def get_body_center_x(landmarks):
    """Calculate the center X position of the body"""
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

    avg_x = (left_shoulder.x + right_shoulder.x) / 2
    return avg_x


def get_arm_positions(landmarks):
    """Get the Y positions of both wrists relative to mid-torso"""
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

    # Calculate mid-torso position (between shoulders and hips)
    mid_torso_y = (left_shoulder.y + right_shoulder.y + left_hip.y + right_hip.y) / 4

    # Calculate relative positions (wrist Y - mid_torso Y)
    # Negative means arm is above mid-torso, positive means below
    left_arm_y = left_wrist.y - mid_torso_y
    right_arm_y = right_wrist.y - mid_torso_y

    return left_arm_y, right_arm_y


def detect_jogging(arm_history):
    """
    Detect jogging by checking if at least one arm is moving up and down a lot
    Returns True if jogging motion is detected
    """
    if len(arm_history) < 8:
        return False

    # Just check if there's a lot of arm movement total
    total_left_movement = 0
    total_right_movement = 0

    for i in range(len(arm_history) - 1):
        left_curr, right_curr = arm_history[i]
        left_next, right_next = arm_history[i + 1]

        # Add up how much each arm is moving
        total_left_movement += abs(left_next - left_curr)
        total_right_movement += abs(right_next - right_curr)

    # If EITHER arm is moving a decent amount, you're jogging!
    if total_left_movement > 0.1 or total_right_movement > 0.1:
        return True

    return False


def detect_jump(current_y, past_y):
    vertical_movement = past_y - current_y

    if (
        vertical_movement > VERTICAL_THRESHOLD
        and not player.is_jumping
        and not player.is_ducking
        and player.jump_cooldown == 0
    ):

        print("🔥 JUMP!")
        player.is_jumping = True
        player.jump_timer = JUMP_DURATION
        player.jump_cooldown = COOLDOWN_DURATION
        return True
    return False


def detect_duck(current_y, past_y):
    vertical_movement = current_y - past_y

    if (
        vertical_movement > VERTICAL_THRESHOLD
        and not player.is_ducking
        and not player.is_jumping
        and player.duck_cooldown == 0
    ):

        print("⬇️ DUCK!")
        player.is_ducking = True
        player.duck_timer = DUCK_DURATION
        player.duck_cooldown = COOLDOWN_DURATION
        return True
    return False


def detect_left_right(current_x, past_x):
    horizontal_movement = current_x - past_x

    if horizontal_movement > HORIZONTAL_THRESHOLD and player.left_cooldown == 0:
        print(f"➡️ MOVE RIGHT!")
        player.is_moving_left = True
        player.left_timer = 15
        player.left_cooldown = COOLDOWN_DURATION
        return "LEFT"

    elif horizontal_movement < -HORIZONTAL_THRESHOLD and player.right_cooldown == 0:
        print(f"⬅️ MOVE LEFT!")
        player.is_moving_right = True
        player.right_timer = 15
        player.right_cooldown = COOLDOWN_DURATION
        return "RIGHT"

    return None


def draw_game_ui(frame, landmarks, h, w):
    for connection in mp_pose.POSE_CONNECTIONS:
        start_idx = connection[0]
        end_idx = connection[1]

        if start_idx <= 10 or end_idx <= 10:
            continue

        start = landmarks[start_idx]
        end = landmarks[end_idx]

        start_point = (int(start.x * w), int(start.y * h))
        end_point = (int(end.x * w), int(end.y * h))

        # Change color if jogging
        color = (0, 255, 255) if player.is_jogging else (0, 255, 0)
        cv2.line(frame, start_point, end_point, color, 3)

    for idx, landmark in enumerate(landmarks):
        if idx <= 10:
            continue
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        color = (255, 255, 0) if player.is_jogging else (0, 0, 255)
        cv2.circle(frame, (x, y), 6, color, -1)

    nose = landmarks[mp_pose.PoseLandmark.NOSE]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

    nose_x = int(nose.x * w)
    nose_y = int(nose.y * h)
    shoulder_width = abs(left_shoulder.x - right_shoulder.x) * w
    head_radius = int(shoulder_width * 0.35)

    head_color = (255, 255, 0) if not player.is_jogging else (0, 255, 255)
    cv2.circle(frame, (nose_x, nose_y), head_radius, head_color, 3)

    y_offset = 150

    if player.is_jumping:
        cv2.putText(
            frame,
            "JUMP!",
            (w // 2 - 120, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.5,
            (0, 255, 0),
            5,
        )
        y_offset += 80

    if player.is_ducking:
        cv2.putText(
            frame,
            "DUCK!",
            (w // 2 - 120, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.5,
            (255, 128, 0),
            5,
        )
        y_offset += 80

    if player.is_jogging:
        cv2.putText(
            frame,
            "JOGGING!",
            (w // 2 - 180, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.5,
            (0, 255, 255),
            5,
        )
        y_offset += 80

    if player.is_moving_left:
        cv2.putText(
            frame,
            "RIGHT!",
            (50, h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.5,
            (255, 0, 255),
            5,
        )

    if player.is_moving_right:
        cv2.putText(
            frame,
            "LEFT!",
            (w - 250, h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.5,
            (0, 255, 255),
            5,
        )

    calories = calorie_tracker.get_calories()
    session_time = calorie_tracker.get_session_time()

    cv2.rectangle(frame, (10, 10), (350, 110), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (350, 110), (0, 255, 0), 2)

    cv2.putText(
        frame,
        f"CALORIES: {calories}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3,
    )
    cv2.putText(
        frame,
        f"TIME: {session_time:.1f}s",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 0),
        2,
    )


print("SUBWAY SURFERS - Motion Control")
print("=" * 50)
print("Controls:")
print("  JUMP: Quick upward movement")
print("  DUCK: Quick downward movement")
print("  MOVE: Move your body left/right")
print("  JOG: Pump your arms alternating up/down")
print("  Q: Quit")
print("=" * 50)

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from camera!")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        display_frame = frame.copy()
        display_frame[:] = 0

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            current_y = get_body_center_y(landmarks)
            current_x = get_body_center_x(landmarks)
            left_arm_y, right_arm_y = get_arm_positions(landmarks)

            position_history.append((current_y, current_x))
            arm_history.append((left_arm_y, right_arm_y))

            if len(position_history) >= 8:
                past_y, past_x = position_history[0]

                detect_jump(current_y, past_y)
                detect_duck(current_y, past_y)
                detect_left_right(current_x, past_x)

            # Detect jogging
            if detect_jogging(arm_history) and not player.is_jogging:
                print("🏃 JOGGING!")
                player.is_jogging = True
                player.jog_timer = JOG_DURATION

            draw_game_ui(display_frame, landmarks, h, w)

            is_moving = (
                player.is_jumping
                or player.is_ducking
                or player.is_moving_left
                or player.is_moving_right
                or player.is_jogging
            )
            calorie_tracker.update(player.is_jumping or player.is_jogging, is_moving)

        else:
            cv2.putText(
                display_frame,
                "STAND IN FRONT OF CAMERA",
                (w // 2 - 250, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

        player.update_timers()

        cv2.imshow("Subway Surfers - Motion Control", display_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("\nGame interrupted by user (Ctrl+C)")
except Exception as e:
    print(f"\nERROR: {e}")
    import traceback

    traceback.print_exc()
finally:
    print("\nCleaning up...")
    cap.release()
    cv2.destroyAllWindows()

    if HAS_CALORIE_TRACKER:
        calorie_tracker.end_session()
        print(f"Game Over!")
        print(f"Final Calories Burned: {calorie_tracker.get_calories()}")
        print(f"Total Time: {calorie_tracker.get_session_time():.1f}s")
    else:
        print("Game Over!")
