import time

# MET values for your activities
IDLE_METS = 1.5      # truly resting
JOGGING_METS = 5.0   # moving/jogging in place
JUMP_METS = 8.0      # jumping

class CalorieTracker:
    """Minimal calorie tracker for IRL Subway Surfers movements"""
    
    def __init__(self, weight_kg: float):
        self.weight_kg = weight_kg
        self.total_calories = 0.0
        self.session_start_time = None
        self.last_update_time = None

    def start_session(self):
        self.session_start_time = time.time()
        self.last_update_time = self.session_start_time
        self.total_calories = 0.0

    def update(self, is_jumping: bool, is_moving: bool):
        """Call this repeatedly with current movement flags"""
        if self.session_start_time is None:
            return  # session not started

        current_time = time.time()
        elapsed_seconds = current_time - self.last_update_time
        self.last_update_time = current_time

        # Determine MET for this frame
        if is_jumping:
            mets = JUMP_METS
        elif is_moving:
            mets = JOGGING_METS
        else:
            mets = IDLE_METS

        # Optional: count only active calories above resting
        active_mets = max(mets - IDLE_METS, 0)

        # Update total calories
        self.total_calories += active_mets * self.weight_kg * (elapsed_seconds / 3600)

    def get_calories(self) -> float:
        return round(self.total_calories, 2)

    def get_session_time(self) -> float:
        if self.session_start_time is None:
            return 0.0
        return time.time() - self.session_start_time
