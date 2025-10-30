import cv2
import numpy as np
import time
import math


# --- Helper Class: Vehicle Tracker ---
class VehicleTracker:
    """
    A simple tracker to assign IDs to vehicles and check if they cross a line.
    """

    # --- MODIFIED: Init now takes crossing_line_x ---
    def __init__(self, max_distance, crossing_line_x):
        self.max_distance = max_distance  # Max pixels a vehicle can move between frames
        self.crossing_line_x = crossing_line_x  # --- MODIFIED ---
        self.active_tracks = {}  # Stores {id: (cx, cy)}
        self.next_id = 0
        self.crossed_ids = set()  # IDs of vehicles that have already crossed

    def update(self, new_centroids):
        """Updates tracker with new centroids, returns count of *new* crossings."""

        crossed_count_this_frame = 0
        new_active_tracks = {}
        matched_new_indices = set()

        # 1. Try to match existing tracks
        for track_id, old_centroid in self.active_tracks.items():

            best_match = None
            min_dist = self.max_distance
            best_index = -1

            for i, new_centroid in enumerate(new_centroids):
                if i in matched_new_indices:
                    continue

                dist = math.hypot(new_centroid[0] - old_centroid[0], new_centroid[1] - old_centroid[1])

                if dist < min_dist:
                    min_dist = dist
                    best_match = new_centroid
                    best_index = i

            # If a match was found
            if best_match:
                new_active_tracks[track_id] = best_match
                matched_new_indices.add(best_index)

                # --- MODIFIED: CHECK FOR X-AXIS LINE CROSSING ---
                old_x = old_centroid[0]
                new_x = best_match[0]

                # Check for crossing in either direction (left or right)
                if (old_x < self.crossing_line_x and new_x >= self.crossing_line_x) or \
                        (old_x > self.crossing_line_x and new_x <= self.crossing_line_x):
                    # Check if we've already counted this ID
                    if track_id not in self.crossed_ids:
                        crossed_count_this_frame += 1
                        self.crossed_ids.add(track_id)

        # 2. Add new, unmatched centroids as new tracks
        for i, new_centroid in enumerate(new_centroids):
            if i not in matched_new_indices:
                new_active_tracks[self.next_id] = new_centroid
                self.next_id += 1

        # 3. Update the active tracks
        self.active_tracks = new_active_tracks
        return crossed_count_this_frame

    def reset_crossed_ids(self):
        """Resets the set of crossed IDs for a new traffic light cycle."""
        self.crossed_ids = set()


# --- Main Application Logic ---

def calculate_green_duration(vehicle_count):
    """
    Calculates green light duration based on vehicle count.
    Logic from your report: T(green) = max(10, min(60, 10 + (N * 5)))
    """
    if vehicle_count == 0:
        return 10  # Minimum green time even for 0 cars

    base_time = 10
    time_per_vehicle = 5
    max_time = 60

    duration = base_time + (vehicle_count * time_per_vehicle)

    if duration > max_time:
        return max_time

    return duration


def main():
    # --- 1. Video and Model Setup ---
    video_file_name = "Traffic_2.mp4"  # <--- IMPORTANT: Change this

    MAX_TRACKING_DISTANCE = 50  # Max pixels to link a vehicle frame-to-frame
    MIN_VEHICLE_AREA = 500  # To filter out noise (as per your report)

    cap = cv2.VideoCapture(video_file_name)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_file_name}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # --- MODIFIED: Set crossing line to the middle of the video width ---
    CROSSING_LINE_X = frame_width // 2

    # Create resizable window
    window_name = "AI Traffic Light Simulation (Press 'q' to quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Create Background Subtractor (as per your report)
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

    # Morphological Kernel (as per your report: 5x5 elliptical)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # --- 2. Traffic Light & Counter Setup ---
    traffic_light_state = "GREEN"  # Start with GREEN
    state_start_time = time.time()

    # Durations
    GREEN_LIGHT_DURATION = 10  # Initial duration
    YELLOW_LIGHT_DURATION = 3  # Fixed
    RED_LIGHT_DURATION = 30  # Fixed

    # Counters
    total_vehicles_crossed = 0
    vehicles_waiting_for_green = 0  # This counts vehicles for the *next* cycle

    # Initialize the tracker
    # --- MODIFIED: Pass CROSSING_LINE_X to the tracker ---
    tracker = VehicleTracker(MAX_TRACKING_DISTANCE, CROSSING_LINE_X)

    # --- 3. Main Loop ---
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video. Resetting...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Clone frame for drawing
        display_frame = frame.copy()

        # --- 4. Vehicle Detection ---

        # a) Apply MOG2
        fg_mask = bg_subtractor.apply(frame)

        # b) Threshold to remove shadows
        _, fg_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)

        # c) Noise Reduction (Morphological Operations)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

        # d) --- REMOVED: Apply ROI Mask section is removed ---

        # e) Find Contours in the *full* frame
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        current_centroids = []
        for cnt in contours:
            # Filter by area
            if cv2.contourArea(cnt) > MIN_VEHICLE_AREA:
                # Get centroid
                M = cv2.moments(cnt)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    current_centroids.append((cx, cy))

                    # --- REMOVED: Draw green box for detected (but untracked) contour ---
                    # (vx, vy, vw, vh) = cv2.boundingRect(cnt)
                    # cv2.rectangle(display_frame, (vx, vy), (vx + vw, vy + vh), (0, 255, 0), 1)

        # --- 5. Update Tracker & Count Crossings ---
        crossed_count_this_frame = tracker.update(current_centroids)

        # Update counts *only* if light is RED (i.e., cars are *waiting*)
        if traffic_light_state == "RED":
            vehicles_waiting_for_green += crossed_count_this_frame

        # Always update total count for display
        total_vehicles_crossed += crossed_count_this_frame

        # --- 6. Traffic Light Logic ---
        current_time = time.time()
        time_in_state = current_time - state_start_time

        if traffic_light_state == "GREEN":
            if time_in_state > GREEN_LIGHT_DURATION:
                traffic_light_state = "YELLOW"
                state_start_time = current_time

        elif traffic_light_state == "YELLOW":
            if time_in_state > YELLOW_LIGHT_DURATION:
                traffic_light_state = "RED"
                state_start_time = current_time
                # Reset the crossed IDs set for the new red cycle
                tracker.reset_crossed_ids()

        elif traffic_light_state == "RED":
            if time_in_state > RED_LIGHT_DURATION:
                # --- THIS IS THE KEY LOGIC ---
                # Calculate next green light duration based on waiting cars
                GREEN_LIGHT_DURATION = calculate_green_duration(vehicles_waiting_for_green)

                # Reset the waiting count
                vehicles_waiting_for_green = 0

                # Switch to Green
                traffic_light_state = "GREEN"
                state_start_time = current_time

        # --- 7. Drawing ---

        # --- REMOVED: Draw ROI Box ---

        # --- MODIFIED: Draw Crossing Line vertically ---
        cv2.line(display_frame, (CROSSING_LINE_X, 0), (CROSSING_LINE_X, frame_height), (0, 0, 255), 2)
        cv2.putText(display_frame, "Crossing Line", (CROSSING_LINE_X + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 2)

        # --- REMOVED: Draw tracked centroids and IDs ---
        # for track_id, (cx, cy) in tracker.active_tracks.items():
        #     cv2.circle(display_frame, (cx, cy), 5, (0, 255, 255), -1) # Yellow dot
        #     cv2.putText(display_frame, f"ID:{track_id}", (cx + 5, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        # Draw UI Panel
        light_panel_x = frame_width - 110
        # Draw lights
        colors = {"RED": (0, 0, 255), "YELLOW": (0, 255, 255), "GREEN": (0, 255, 0)}
        cv2.circle(display_frame, (light_panel_x + 50, 50), 20, (50, 50, 50), -1)  # Red bg
        cv2.circle(display_frame, (light_panel_x + 50, 100), 20, (50, 50, 50), -1)  # Yellow bg
        cv2.circle(display_frame, (light_panel_x + 50, 150), 20, (50, 50, 50), -1)  # Green bg

        # Draw active light
        if traffic_light_state == "RED":
            cv2.circle(display_frame, (light_panel_x + 50, 50), 20, colors["RED"], -1)
        elif traffic_light_state == "YELLOW":
            cv2.circle(display_frame, (light_panel_x + 50, 100), 20, colors["YELLOW"], -1)
        elif traffic_light_state == "GREEN":
            cv2.circle(display_frame, (light_panel_x + 50, 150), 20, colors["GREEN"], -1)

        # Draw Text
        timer_text = f"{int(time_in_state)}s"
        cv2.putText(display_frame, timer_text, (light_panel_x, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(display_frame, f"Total Crossed: {total_vehicles_crossed}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)
        cv2.putText(display_frame, f"Vehicles Waiting: {vehicles_waiting_for_green}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(display_frame, f"Next Green: {GREEN_LIGHT_DURATION}s", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2)

        # --- 8. Display Frame ---
        cv2.imshow(window_name, display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- 9. Cleanup ---
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


