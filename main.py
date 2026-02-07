from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel
from PyQt5.QtCore import Qt, QRectF, QTimer, QPropertyAnimation, QEasingCurve, QPoint, QRect
from PyQt5.QtGui import QPainter, QBrush, QColor, QPainterPath, QRegion, QMovie, QLinearGradient, QImage, QPixmap
import sys
import cv2
import mediapipe as mp
from collections import deque
from CalorieTracker import CalorieTracker
from pynput.keyboard import Controller, Key


class RoundedWindow(QWidget):
    def __init__(self):
        super().__init__()

        # Window settings
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)

        # Size and position
        screen = QApplication.primaryScreen().geometry()
        available = QApplication.primaryScreen().availableGeometry()
        self.width_val = 580  # Wider for better layout
        self.height_val = 320  # Taller for better spacing
        x = screen.width() - self.width_val - 10
        y = available.y() + 10
        self.setGeometry(x, y, self.width_val, self.height_val)

        # Title text
        self.title_text = "I Like to Move It"

        # BIG CENTERED TITLE for intro
        self.big_title = QLabel(self.title_text, self)
        self.big_title.setStyleSheet("""
            QLabel {
                color: #d6fd51;
                font-size: 28px;
                font-weight: bold;
                background: transparent;
            }
        """)
        self.big_title.setAlignment(Qt.AlignCenter)
        # Make the label width smaller so it doesn't extend behind the running man
        self.big_title.setGeometry(0, self.height_val // 2 - 30, self.width_val - 55, 60)

        # Overlay to hide the big title
        self.big_overlay = QLabel(self)
        self.big_overlay.setStyleSheet("background-color: black;")
        self.big_overlay.setGeometry(0, self.height_val // 2 - 30, self.width_val - 55, 60)

        # Running guy GIF (starts center-left, bigger)
        self.running_guy = QLabel(self)
        self.movie = QMovie("./running_man.gif")
        self.running_guy.setMovie(self.movie)
        self.running_guy.setScaledContents(True)
        self.running_guy.setGeometry(30, self.height_val // 2 - 20, 40, 40)  # Big at start
        self.running_guy.raise_()
        self.movie.start()

        # Track animation state
        self.should_stop = False
        self.movie.frameChanged.connect(self.check_frame)

        # Animation 1: Running guy moves across to reveal title
        self.reveal_animation = QPropertyAnimation(self.running_guy, b"pos")
        self.reveal_animation.setDuration(2900)
        self.reveal_animation.setStartValue(self.running_guy.pos())
        self.reveal_animation.setEasingCurve(QEasingCurve.InOutQuad)
        self.reveal_animation.finished.connect(self.start_fade_and_shrink)

        # Animation 2: Overlay shrinks to reveal title (slower than running guy)
        self.overlay_reveal = QPropertyAnimation(self.big_overlay, b"geometry")
        self.overlay_reveal.setDuration(3300)
        self.overlay_reveal.setStartValue(self.big_overlay.geometry())
        self.overlay_reveal.setEasingCurve(QEasingCurve.InOutQuad)

        # Animation 3: Title fades out
        self.title_fade = QPropertyAnimation(self.big_title, b"windowOpacity")
        self.title_fade.setDuration(800)
        self.title_fade.setStartValue(1.0)
        self.title_fade.setEndValue(0.0)

        # Animation 4: Guy moves to top left
        self.move_to_corner = QPropertyAnimation(self.running_guy, b"geometry")
        self.move_to_corner.setDuration(1000)
        self.move_to_corner.setEasingCurve(QEasingCurve.InOutQuad)
        self.move_to_corner.finished.connect(self.finish_intro)

        # Close button (hidden at start)
        self.close_btn = QPushButton("✕", self)
        self.close_btn.setGeometry(self.width_val - 35, 15, 20, 20)
        self.close_btn.setStyleSheet("""
            QPushButton {
                background-color: #d6fd51;
                color: black;
                border: none;
                border-radius: 6px;
                font-size: 18px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #c5ec40;
            }
        """)
        self.close_btn.clicked.connect(self.close)
        self.close_btn.hide()  # Hidden during intro

        # Gray circle for calorie counter (left side, hidden at start)
        self.calorie_circle = QLabel(self)
        circle_size = 140
        circle_x = 20
        circle_y = 60
        self.calorie_circle.setGeometry(circle_x, circle_y, circle_size, circle_size)
        self.calorie_circle.setStyleSheet(f"""
            QLabel {{
                background-color: transparent;
                border: 3px solid #404040;
                border-radius: {circle_size // 2}px;
            }}
        """)
        self.calorie_circle.hide()

        # Calorie number (big yellow text)
        self.calories = 0
        self.calorie_number = QLabel(str(self.calories), self)
        self.calorie_number.setGeometry(circle_x, circle_y + 30, circle_size, 50)
        self.calorie_number.setAlignment(Qt.AlignCenter)
        self.calorie_number.setStyleSheet("""
            QLabel {
                color: #d6fd51;
                font-size: 42px;
                font-weight: bold;
                background: transparent;
            }
        """)
        self.calorie_number.hide()

        # "Calories Burned" label (small gray text)
        self.calorie_label = QLabel("Calories\nBurned", self)
        self.calorie_label.setGeometry(circle_x, circle_y + 80, circle_size, 45)
        self.calorie_label.setAlignment(Qt.AlignCenter)
        self.calorie_label.setStyleSheet("""
            QLabel {
                color: #909090;
                font-size: 12px;
                background: transparent;
                line-height: 1.3;
            }
        """)
        self.calorie_label.hide()

        # Pose outline display (right side, hidden at start)
        self.pose_display = QLabel(self)
        pose_width = 240
        pose_height = 260
        pose_x = self.width_val - pose_width - 20
        pose_y = 45
        self.pose_display.setGeometry(pose_x, pose_y, pose_width, pose_height)
        self.pose_display.setStyleSheet("""
            QLabel {
                background-color: rgba(10, 10, 10, 150);
                border: 3px solid #505050;
                border-radius: 12px;
            }
        """)
        self.pose_display.setScaledContents(True)
        self.pose_display.hide()

        # Pose display title (hidden at start)
        self.pose_title = QLabel("POSE TRACKING", self)
        self.pose_title.setGeometry(pose_x, 15, pose_width, 25)
        self.pose_title.setAlignment(Qt.AlignCenter)
        self.pose_title.setStyleSheet("""
            QLabel {
                color: #d6fd51;
                font-size: 14px;
                font-weight: bold;
                background: transparent;
            }
        """)
        self.pose_title.hide()

        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

        # Initialize video capture
        self.cap = cv2.VideoCapture(0)

        # Initialize keyboard controller for emulating key presses
        self.keyboard = Controller()

        # Initialize CalorieTracker
        weight_kg = 70  # Default weight, adjust as needed
        self.calorie_tracker = CalorieTracker(weight_kg)
        self.calorie_tracker.start_session()

        # Position history for movement detection
        self.position_history = deque(maxlen=10)
        self.arm_history = deque(maxlen=10)

        # Movement detection variables
        self.jump_threshold = 0.05
        self.is_jumping = False
        self.jump_timer = 0

        self.left_right_threshold = 0.08
        self.is_moving_left = False
        self.is_moving_right = False
        self.left_timer = 0
        self.right_timer = 0

        self.duck_threshold = 0.05
        self.is_ducking = False
        self.duck_timer = 0

        self.is_jogging = False
        self.jog_timer = 0

        # Cooldowns
        self.left_cooldown = 0
        self.right_cooldown = 0
        self.jump_cooldown = 0
        self.duck_cooldown = 0
        self.post_jump_cooldown = 0  # Prevents duck detection right after landing
        self.post_duck_cooldown = 0  # Prevents jump detection right after standing up

        # Timer for video capture (initially stopped)
        self.video_timer = QTimer()
        self.video_timer.timeout.connect(self.update_pose_frame)

        # Timer for calorie updates
        self.calorie_update_timer = QTimer()
        self.calorie_update_timer.timeout.connect(self.update_calorie_display)

        # Start animation after window shows
        QTimer.singleShot(500, self.start_intro)

    def start_intro(self):
        # Stage 1: Guy runs across, revealing big centered title

        # Move guy to end position (beyond the text area)
        end_x = self.width_val - 65
        self.reveal_animation.setEndValue(QPoint(end_x, self.height_val // 2 - 20))
        self.reveal_animation.start()

        # Shrink overlay to reveal title from right to left
        # The overlay and text both end at width_val - 55, so reveal fully within that space
        self.overlay_reveal.setEndValue(QRect(self.width_val - 55, self.height_val // 2 - 30, 0, 60))
        self.overlay_reveal.start()

    def start_fade_and_shrink(self):
        # Stage 2: Title fades out, guy shrinks and moves to top left
        self.should_stop = True  # Stop GIF on next frame 0

        # Fade out the big title
        self.big_title.setStyleSheet("""
            QLabel {
                color: #d6fd51;
                font-size: 28px;
                font-weight: bold;
                background: transparent;
            }
        """)

        # Create fade effect using timer
        self.fade_opacity = 1.0
        self.fade_timer = QTimer()
        self.fade_timer.timeout.connect(self.fade_step)
        self.fade_timer.start(30)

        # Wait a bit, then move guy to corner
        QTimer.singleShot(800, self.move_guy_to_corner)

    def fade_step(self):
        self.fade_opacity -= 0.05
        if self.fade_opacity <= 0:
            self.fade_timer.stop()
            self.big_title.hide()
            self.big_overlay.hide()
        else:
            # Update title opacity
            opacity_int = int(self.fade_opacity * 255)
            self.big_title.setStyleSheet(f"""
                QLabel {{
                    color: rgba(214, 253, 81, {opacity_int});
                    font-size: 28px;
                    font-weight: bold;
                    background: transparent;
                }}
            """)

    def move_guy_to_corner(self):
        # Move and shrink guy to top left corner
        self.move_to_corner.setStartValue(self.running_guy.geometry())
        self.move_to_corner.setEndValue(QRect(15, 15, 24, 24))  # Small, top left
        self.move_to_corner.start()

    def finish_intro(self):
        # Show close button
        self.close_btn.show()
        self.close_btn.raise_()

        # Show calorie counter elements
        self.calorie_circle.show()
        self.calorie_number.show()
        self.calorie_label.show()
        self.calorie_number.raise_()
        self.calorie_label.raise_()

        # Show pose display and start video capture
        self.pose_title.show()
        self.pose_title.raise_()
        self.pose_display.show()
        self.pose_display.raise_()
        self.video_timer.start(33)  # ~30 FPS
        self.calorie_update_timer.start(100)  # Update calories 10 times per second

    def check_frame(self, frame_number):
        # Stop on frame 0 after animation is done
        if self.should_stop and frame_number == 0:
            self.movie.stop()
            self.movie.jumpToFrame(0)

    def update_calories(self, value):
        """Update the calorie counter display"""
        self.calories = value
        self.calorie_number.setText(str(self.calories))

    # ============================================
    # Movement Detection Callback Methods
    # ============================================
    # These methods are called when movements are detected.
    # They simulate keyboard arrow key presses system-wide.

    def on_jump_detected(self):
        """Called when a jump is detected - Presses UP arrow key"""
        print("🔥 JUMP DETECTED! → Pressing UP arrow")
        self.keyboard.press(Key.up)
        self.keyboard.release(Key.up)

    def on_duck_detected(self):
        """Called when a duck is detected - Presses DOWN arrow key"""
        print("⬇️ DUCK DETECTED! → Pressing DOWN arrow")
        self.keyboard.press(Key.down)
        self.keyboard.release(Key.down)

    def on_move_left_detected(self):
        """Called when left movement is detected - Presses LEFT arrow key"""
        print("⬅️ MOVE LEFT DETECTED! → Pressing LEFT arrow")
        self.keyboard.press(Key.left)
        self.keyboard.release(Key.left)

    def on_move_right_detected(self):
        """Called when right movement is detected - Presses RIGHT arrow key"""
        print("➡️ MOVE RIGHT DETECTED! → Pressing RIGHT arrow")
        self.keyboard.press(Key.right)
        self.keyboard.release(Key.right)

    def on_jogging_detected(self):
        """Called when jogging is detected"""
        print("🏃 JOGGING DETECTED!")
        # No key press for jogging - you can add custom logic here if needed

    def on_jump_end(self):
        """Called when jump ends"""
        # No action on end

    def on_duck_end(self):
        """Called when duck ends"""
        # No action on end

    def on_move_left_end(self):
        """Called when left movement ends"""
        # No action on end

    def on_move_right_end(self):
        """Called when right movement ends"""
        # No action on end

    def on_jogging_end(self):
        """Called when jogging ends"""
        # No action on end

    # ============================================

    def get_body_center_y(self, landmarks):
        """Calculate the center Y position of the torso"""
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
        avg_y = (left_shoulder.y + right_shoulder.y + left_hip.y + right_hip.y) / 4
        return avg_y

    def get_body_center_x(self, landmarks):
        """Calculate the center X position of the body"""
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        avg_x = (left_shoulder.x + right_shoulder.x) / 2
        return avg_x

    def get_arm_positions(self, landmarks):
        """Get the Y positions of both wrists relative to mid-torso"""
        left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]

        mid_torso_y = (left_shoulder.y + right_shoulder.y + left_hip.y + right_hip.y) / 4
        left_arm_y = left_wrist.y - mid_torso_y
        right_arm_y = right_wrist.y - mid_torso_y
        return left_arm_y, right_arm_y

    def detect_jogging(self):
        """Detect jogging by checking arm movement"""
        if len(self.arm_history) < 8:
            return False

        total_left_movement = 0
        total_right_movement = 0

        for i in range(len(self.arm_history) - 1):
            left_curr, right_curr = self.arm_history[i]
            left_next, right_next = self.arm_history[i + 1]
            total_left_movement += abs(left_next - left_curr)
            total_right_movement += abs(right_next - right_curr)

        return total_left_movement > 0.1 or total_right_movement > 0.1

    def update_timers(self):
        """Update all movement timers"""
        if self.jump_timer > 0:
            self.jump_timer -= 1
        else:
            if self.is_jumping:
                self.on_jump_end()
                # Set post-jump cooldown to prevent duck detection after landing
                self.post_jump_cooldown = 60  # ~2 seconds - even bigger lock to prevent accidental duck
                print(f"[DEBUG] Jump ended, post_jump_cooldown set to 60")
            self.is_jumping = False

        if self.duck_timer > 0:
            self.duck_timer -= 1
        else:
            if self.is_ducking:
                self.on_duck_end()
                # Set post-duck cooldown to prevent jump detection after standing up
                self.post_duck_cooldown = 60  # ~2 seconds - even bigger lock to prevent accidental jump
                print(f"[DEBUG] Duck ended, post_duck_cooldown set to 60")
            self.is_ducking = False

        if self.left_timer > 0:
            self.left_timer -= 1
        else:
            if self.is_moving_left:
                self.on_move_left_end()
            self.is_moving_left = False

        if self.right_timer > 0:
            self.right_timer -= 1
        else:
            if self.is_moving_right:
                self.on_move_right_end()
            self.is_moving_right = False

        if self.jog_timer > 0:
            self.jog_timer -= 1
        else:
            if self.is_jogging:
                self.on_jogging_end()
            self.is_jogging = False

        if self.left_cooldown > 0:
            self.left_cooldown -= 1
        if self.right_cooldown > 0:
            self.right_cooldown -= 1
        if self.jump_cooldown > 0:
            self.jump_cooldown -= 1
        if self.duck_cooldown > 0:
            self.duck_cooldown -= 1
        if self.post_jump_cooldown > 0:
            self.post_jump_cooldown -= 1
        if self.post_duck_cooldown > 0:
            self.post_duck_cooldown -= 1

    def update_pose_frame(self):
        """Capture and process video frame, update pose display"""
        ret, frame = self.cap.read()
        if not ret:
            return

        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        # Create black canvas for pose outline
        display_frame = frame.copy()
        display_frame[:] = 0

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Track body position
            current_y = self.get_body_center_y(landmarks)
            current_x = self.get_body_center_x(landmarks)
            left_arm_y, right_arm_y = self.get_arm_positions(landmarks)

            self.position_history.append((current_y, current_x))
            self.arm_history.append((left_arm_y, right_arm_y))

            # Detect movements
            if len(self.position_history) >= 8:
                past_y, past_x = self.position_history[0]

                # Jump detection
                vertical_movement = past_y - current_y
                if (vertical_movement > self.jump_threshold and not self.is_jumping
                    and not self.is_ducking and self.jump_cooldown == 0
                    and self.post_duck_cooldown == 0):  # Don't detect jump right after standing up
                    self.is_jumping = True
                    self.jump_timer = 15
                    self.jump_cooldown = 10
                    self.on_jump_detected()

                # Duck detection
                vertical_movement_down = current_y - past_y
                if (vertical_movement_down > self.duck_threshold and not self.is_ducking
                    and not self.is_jumping and self.duck_cooldown == 0
                    and self.post_jump_cooldown == 0):  # Don't detect duck right after landing
                    print(f"[DEBUG] Duck detected! vertical_movement_down={vertical_movement_down:.3f}, post_jump_cooldown={self.post_jump_cooldown}")
                    self.is_ducking = True
                    self.duck_timer = 20
                    self.duck_cooldown = 10
                    self.on_duck_detected()

                # Left/Right detection
                horizontal_movement = current_x - past_x
                if horizontal_movement > self.left_right_threshold and self.left_cooldown == 0:
                    self.is_moving_left = True
                    self.left_timer = 15
                    self.left_cooldown = 10
                    self.on_move_right_detected()
                elif horizontal_movement < -self.left_right_threshold and self.right_cooldown == 0:
                    self.is_moving_right = True
                    self.right_timer = 15
                    self.right_cooldown = 10
                    self.on_move_left_detected()

            # Jogging detection
            if self.detect_jogging() and not self.is_jogging:
                self.is_jogging = True
                self.jog_timer = 25
                self.on_jogging_detected()

            # Draw pose skeleton (exclude face landmarks)
            for connection in self.mp_pose.POSE_CONNECTIONS:
                start_idx = connection[0]
                end_idx = connection[1]

                # Skip face connections
                if start_idx <= 10 or end_idx <= 10:
                    continue

                start = landmarks[start_idx]
                end = landmarks[end_idx]

                start_point = (int(start.x * w), int(start.y * h))
                end_point = (int(end.x * w), int(end.y * h))

                # Change color based on movement
                color = (0, 255, 255) if self.is_jogging else (0, 255, 0)
                cv2.line(display_frame, start_point, end_point, color, 3)

            # Draw pose landmarks
            for idx, landmark in enumerate(landmarks):
                if idx <= 10:  # Skip face landmarks
                    continue
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                color = (255, 255, 0) if self.is_jogging else (0, 0, 255)
                cv2.circle(display_frame, (x, y), 6, color, -1)

            # Draw head circle
            nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]

            nose_x = int(nose.x * w)
            nose_y = int(nose.y * h)
            shoulder_width = abs(left_shoulder.x - right_shoulder.x) * w
            head_radius = int(shoulder_width * 0.35)

            head_color = (255, 255, 0) if not self.is_jogging else (0, 255, 255)
            cv2.circle(display_frame, (nose_x, nose_y), head_radius, head_color, 3)

        # Convert OpenCV image to QPixmap
        rgb_image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)

        # Display in pose_display label
        self.pose_display.setPixmap(pixmap)

        # Update timers
        self.update_timers()

        # Update calorie tracker
        is_moving = (self.is_jumping or self.is_ducking or
                    self.is_moving_left or self.is_moving_right or self.is_jogging)
        self.calorie_tracker.update(self.is_jumping or self.is_jogging, is_moving)

    def update_calorie_display(self):
        """Update the calorie display with current value"""
        calories = int(self.calorie_tracker.get_calories())
        self.update_calories(calories)

    def closeEvent(self, event):
        """Clean up resources when closing"""
        self.video_timer.stop()
        self.calorie_update_timer.stop()
        if self.cap is not None:
            self.cap.release()
        event.accept()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Create rounded rectangle path
        path = QPainterPath()
        path.addRoundedRect(QRectF(0, 0, self.width(), self.height()), 20, 20)

        # Create black gradient
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0, QColor("#000000"))
        gradient.setColorAt(0.5, QColor("#0a0a0a"))
        gradient.setColorAt(1, QColor("#000000"))

        # Fill with gradient
        painter.fillPath(path, QBrush(gradient))

        # Set the window mask for true rounded corners
        region = QRegion(path.toFillPolygon().toPolygon())
        self.setMask(region)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = RoundedWindow()
    window.show()
    sys.exit(app.exec_())