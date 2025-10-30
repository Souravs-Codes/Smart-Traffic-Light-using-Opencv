import cv2
import numpy as np
import time
import math



class VehicleTracker:

    def __init__(self, max_distance, crossing_line_y):
        self.max_distance = max_distance
        self.crossing_line_y = crossing_line_y
        self.active_tracks = {}
        self.next_id = 0
        self.crossed_ids = set()

    def update(self, new_centroids):

        crossed_count_this_frame = 0
        new_active_tracks = {}
        matched_new_indices = set()


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


            if best_match:
                new_active_tracks[track_id] = best_match
                matched_new_indices.add(best_index)


                old_y = old_centroid[1]
                new_y = best_match[1]


                if (old_y < self.crossing_line_y and new_y >= self.crossing_line_y):
                    # Check if we've already counted this ID
                    if track_id not in self.crossed_ids:
                        crossed_count_this_frame += 1
                        self.crossed_ids.add(track_id)


        for i, new_centroid in enumerate(new_centroids):
            if i not in matched_new_indices:
                new_active_tracks[self.next_id] = new_centroid
                self.next_id += 1


        self.active_tracks = new_active_tracks
        return crossed_count_this_frame

    def reset_crossed_ids(self):

        self.crossed_ids = set()




def calculate_green_duration(vehicle_count):

    if vehicle_count == 0:
        return 0

    base_time = 2
    time_per_vehicle = 2
    max_time = 20

    duration = base_time + (vehicle_count * time_per_vehicle)

    if duration > max_time:
        return max_time

    return duration


def main():

    video_file_name = "Traffic_8.mp4"
    MAX_TRACKING_DISTANCE = 14.8
    MIN_VEHICLE_AREA = 500

    cap = cv2.VideoCapture(video_file_name)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_file_name}")
        return

    #Video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


    CROSSING_LINE_Y = frame_height // 2

    #UI
    base_dimension = max(frame_width, frame_height)  #
    FONT_SCALE = base_dimension / 1200.0
    FONT_THICKNESS = max(1, int(FONT_SCALE * 2))


    UI_MARGIN = int(base_dimension * 0.015)
    UI_LINE_HEIGHT = int(base_dimension * 0.035)


    LIGHT_PANEL_X = frame_width - int(frame_width * 0.1)
    LIGHT_RADIUS = int(base_dimension * 0.025)
    LIGHT_Y_RED = int(frame_height * 0.08)
    LIGHT_Y_YELLOW = LIGHT_Y_RED + int(base_dimension * 0.07)
    LIGHT_Y_GREEN = LIGHT_Y_YELLOW + int(base_dimension * 0.07)
    LIGHT_TIMER_Y = LIGHT_Y_GREEN + int(base_dimension * 0.07)



    window_name = "AI Traffic Light Simulation (Press 'q' to quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Background Subtractor
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=10, detectShadows=True)


    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


    traffic_light_state = "GREEN"
    state_start_time = time.time()

    # Durations
    GREEN_LIGHT_DURATION = 2
    YELLOW_LIGHT_DURATION = 3
    RED_LIGHT_DURATION = 5

    # Counters
    total_vehicles_crossed = 0
    vehicles_waiting_for_green = 0



    tracker = VehicleTracker(MAX_TRACKING_DISTANCE, CROSSING_LINE_Y)


    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video. Resetting...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue


        display_frame = frame.copy()




        fg_mask = bg_subtractor.apply(frame)


        _, fg_mask = cv2.threshold(fg_mask, 170, 255, cv2.THRESH_BINARY)


        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)




        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        current_centroids = []
        for cnt in contours:

            if cv2.contourArea(cnt) > MIN_VEHICLE_AREA:

                M = cv2.moments(cnt)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    current_centroids.append((cx, cy))




        crossed_count_this_frame = tracker.update(current_centroids)

        if traffic_light_state == "RED":
            vehicles_waiting_for_green += crossed_count_this_frame


        total_vehicles_crossed += crossed_count_this_frame

        # Traffic Light Logic
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
                tracker.reset_crossed_ids()

        elif traffic_light_state == "RED":
            if time_in_state > RED_LIGHT_DURATION:
                GREEN_LIGHT_DURATION = calculate_green_duration(vehicles_waiting_for_green)
                vehicles_waiting_for_green = 0
                traffic_light_state = "GREEN"
                state_start_time = current_time

        cv2.line(display_frame, (0, CROSSING_LINE_Y), (frame_width, CROSSING_LINE_Y), (0, 0, 255), FONT_THICKNESS)
        cv2.putText(display_frame, "Crossing Line", (UI_MARGIN, CROSSING_LINE_Y - UI_MARGIN), cv2.FONT_HERSHEY_SIMPLEX,
                    FONT_SCALE * 0.8,
                    (0, 0, 255), FONT_THICKNESS)



        colors = {"RED": (0, 0, 255), "YELLOW": (0, 255, 255), "GREEN": (0, 255, 0)}
        cv2.circle(display_frame, (LIGHT_PANEL_X, LIGHT_Y_RED), LIGHT_RADIUS, (50, 50, 50), -1)  # Red bg
        cv2.circle(display_frame, (LIGHT_PANEL_X, LIGHT_Y_YELLOW), LIGHT_RADIUS, (50, 50, 50), -1)  # Yellow bg
        cv2.circle(display_frame, (LIGHT_PANEL_X, LIGHT_Y_GREEN), LIGHT_RADIUS, (50, 50, 50), -1)  # Green bg


        if traffic_light_state == "RED":
            cv2.circle(display_frame, (LIGHT_PANEL_X, LIGHT_Y_RED), LIGHT_RADIUS, colors["RED"], -1)
        elif traffic_light_state == "YELLOW":
            cv2.circle(display_frame, (LIGHT_PANEL_X, LIGHT_Y_YELLOW), LIGHT_RADIUS, colors["YELLOW"], -1)
        elif traffic_light_state == "GREEN":
            cv2.circle(display_frame, (LIGHT_PANEL_X, LIGHT_Y_GREEN), LIGHT_RADIUS, colors["GREEN"], -1)


        timer_text = f"{int(time_in_state)}s"
        cv2.putText(display_frame, timer_text, (LIGHT_PANEL_X - LIGHT_RADIUS, LIGHT_TIMER_Y), cv2.FONT_HERSHEY_SIMPLEX,
                    FONT_SCALE, (255, 255, 255), FONT_THICKNESS)

        cv2.putText(display_frame, f"Total Crossed: {total_vehicles_crossed}", (UI_MARGIN, UI_LINE_HEIGHT * 1),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE,
                    (255, 255, 255), FONT_THICKNESS)
        cv2.putText(display_frame, f"Vehicles Waiting: {vehicles_waiting_for_green}", (UI_MARGIN, UI_LINE_HEIGHT * 2),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 255, 255), FONT_THICKNESS)
        cv2.putText(display_frame, f"Next Green: {GREEN_LIGHT_DURATION}s", (UI_MARGIN, UI_LINE_HEIGHT * 3),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE,
                    (0, 255, 0), FONT_THICKNESS)

        cv2.imshow(window_name, display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()



