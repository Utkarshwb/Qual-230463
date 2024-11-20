from ultralytics import YOLO
import cv2
import time
import os
import numpy as np
from collections import defaultdict


class VehicleTracker:
    def __init__(self, distance_pixels_per_meter=30):
        self.tracks = defaultdict(list)
        self.vehicle_speeds = {}
        self.next_id = 0
        self.distance_pixels_per_meter = distance_pixels_per_meter

    def calculate_center(self, box):
        x1, y1, x2, y2 = map(int, box)
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))

    def calculate_speed(self, positions, fps):
        if len(positions) < 2:
            return 0

        # Calculate distance in pixels between last two positions
        pos1, pos2 = positions[-2:]
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        distance_pixels = np.sqrt(dx ** 2 + dy ** 2)

        # Convert pixels to meters using calibration factor
        distance_meters = distance_pixels / self.distance_pixels_per_meter

        # Calculate time between frames
        time_seconds = 1 / fps

        # Calculate speed (meters per second)
        speed_mps = distance_meters / time_seconds

        # Convert to km/h
        speed_kmh = speed_mps * 3.6

        # Smooth the speed value
        return min(speed_kmh, 150)  # Cap at 150 km/h to filter outliers

    def update_tracks(self, detections, fps):
        current_detections = []

        # Process new detections
        for box in detections:
            center = self.calculate_center(box.xyxy[0])
            confidence = box.conf[0]
            class_id = int(box.cls[0])  # Convert to int
            current_detections.append((center, confidence, class_id))

        # If there are no existing tracks, create new ones
        if not self.tracks:
            for center, conf, class_id in current_detections:
                self.tracks[self.next_id] = [(center, time.time(), class_id)]
                self.next_id += 1
            return

        # Match new detections with existing tracks
        unmatched_detections = current_detections.copy()
        matched_tracks = set()

        for track_id, track_history in self.tracks.items():
            if not track_history:
                continue

            last_pos = track_history[-1][0]

            # Find the closest detection to this track
            min_dist = float('inf')
            best_match = None

            for i, (center, conf, class_id) in enumerate(unmatched_detections):
                dist = np.sqrt((last_pos[0] - center[0]) ** 2 + (last_pos[1] - center[1]) ** 2)
                if dist < min_dist and dist < 100:  # 100 pixels threshold for matching
                    min_dist = dist
                    best_match = i

            if best_match is not None:
                center, conf, class_id = unmatched_detections[best_match]
                self.tracks[track_id].append((center, time.time(), class_id))
                matched_tracks.add(track_id)
                del unmatched_detections[best_match]

        # Create new tracks for unmatched detections
        for center, conf, class_id in unmatched_detections:
            self.tracks[self.next_id] = [(center, time.time(), class_id)]
            self.next_id += 1

        # Update speeds for all active tracks
        for track_id in matched_tracks:
            positions = [pos for pos, _, _ in self.tracks[track_id]]
            self.vehicle_speeds[track_id] = self.calculate_speed(positions, fps)

        # Remove old tracks
        current_time = time.time()
        old_tracks = [track_id for track_id, track_history in self.tracks.items()
                      if current_time - track_history[-1][1] > 1.0]
        for track_id in old_tracks:
            del self.tracks[track_id]
            if track_id in self.vehicle_speeds:
                del self.vehicle_speeds[track_id]


def run_yolo_detection(source, model_path='best.pt', conf_threshold=0.15, save_output=True):
    # Load the YOLO model
    model = YOLO(model_path)

    # Initialize video capture
    if isinstance(source, int):
        cap = cv2.VideoCapture(source)
        save_output = False
    else:
        if not os.path.exists(source):
            print(f"Error: Video file '{source}' not found")
            return
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Error: Could not open video source")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize video writer if saving output
    if save_output and not isinstance(source, int):
        output_path = f'output_{os.path.basename(source)}'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Initialize vehicle tracker
    tracker = VehicleTracker()

    print("Press 'q' to quit")

    frame_count = 0
    vehicle_counts = defaultdict(int)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if not isinstance(source, int):
            progress = (frame_count / total_frames) * 100
            print(f"\rProgress: {progress:.1f}%", end='')

        start_time = time.time()

        # Run inference on the frame
        results = model(frame, conf=conf_threshold)[0]

        # Update tracker
        tracker.update_tracks(results.boxes, fps)

        # Reset frame counters
        frame_vehicle_counts = defaultdict(int)

        # Calculate processing FPS
        processing_fps = 1 / (time.time() - start_time)

        # Draw detection results and speeds on the frame
        for track_id, track_history in tracker.tracks.items():
            if not track_history:
                continue

            # Get the latest position and class
            current_pos, _, class_id = track_history[-1]
            class_name = model.names[class_id]

            # Update vehicle counts
            frame_vehicle_counts[class_name] += 1

            # Get speed
            speed = tracker.vehicle_speeds.get(track_id, 0)

            # Draw vehicle path
            if len(track_history) > 1:
                points = np.array([pos for pos, _, _ in track_history], np.int32)
                points = points.reshape((-1, 1, 2))
                cv2.polylines(frame, [points], False, (0, 255, 255), 2)

            # Draw current position and speed
            x, y = current_pos

            # Draw circle at current position
            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

            # Draw speed and ID above the circle with background
            speed_text = f"ID:{track_id} {class_name} {speed:.1f}km/h"

            # Get text size for background rectangle
            (text_width, text_height), _ = cv2.getTextSize(
                speed_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )

            # Draw background rectangle
            cv2.rectangle(frame,
                          (int(x) - 5, int(y) - text_height - 25),
                          (int(x) + text_width + 5, int(y) - 10),
                          (0, 0, 0), -1)

            # Draw text
            cv2.putText(frame, speed_text,
                        (int(x), int(y) - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Update maximum counts
        for vehicle_type, count in frame_vehicle_counts.items():
            vehicle_counts[vehicle_type] = max(vehicle_counts[vehicle_type], count)

        # Draw FPS and counts
        y_offset = 30
        cv2.putText(frame, f'FPS: {processing_fps:.1f}',
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display vehicle counts
        for vehicle_type, count in vehicle_counts.items():
            y_offset += 40
            cv2.putText(frame, f'{vehicle_type}: {count}',
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display total vehicles
        y_offset += 40
        total_vehicles = sum(vehicle_counts.values())
        cv2.putText(frame, f'Total: {total_vehicles}',
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('YOLO Vehicle Detection', frame)

        if save_output and not isinstance(source, int):
            out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Print final counts
    print("\n\nFinal Vehicle Counts:")
    for vehicle_type, count in vehicle_counts.items():
        print(f"{vehicle_type}: {count}")
    print(f"Total Vehicles: {total_vehicles}")

    cap.release()
    if save_output and not isinstance(source, int):
        out.release()
    cv2.destroyAllWindows()

    if not isinstance(source, int):
        print("\nProcessing complete!")
        if save_output:
            print(f"Output saved as: {output_path}")


if __name__ == "__main__":
    run_yolo_detection('speedtracker.mp4')