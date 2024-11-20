from ultralytics import YOLO
import cv2
import time
import os


def run_yolo_detection(source, model_path='best.pt', conf_threshold=0.15, save_output=True):
    """
    Run YOLO detection on webcam or video file
    Args:
        source: Integer for webcam (0 for default) or string path for video file
        model_path: Path to YOLO model
        conf_threshold: Confidence threshold for detections
        save_output: Whether to save the output video (only applies to video files)
    """
    # Load the YOLO model
    model = YOLO(model_path)

    # Initialize video capture
    if isinstance(source, int):
        cap = cv2.VideoCapture(source)  # Webcam
        save_output = False  # Don't save output for webcam
    else:
        if not os.path.exists(source):
            print(f"Error: Video file '{source}' not found")
            return
        cap = cv2.VideoCapture(source)  # Video file

    # Check if video/webcam opened successfully
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

    print("Press 'q' to quit")

    frame_count = 0
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("End of video file or error reading frame")
            break

        frame_count += 1
        if not isinstance(source, int):  # Show progress for video files
            progress = (frame_count / total_frames) * 100
            print(f"\rProgress: {progress:.1f}%", end='')

        # Start time to calculate FPS
        start_time = time.time()

        # Run inference on the frame
        results = model(frame, conf=conf_threshold)[0]

        # Calculate processing FPS
        processing_fps = 1 / (time.time() - start_time)

        # Draw detection results on the frame
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            confidence = box.conf[0]
            class_id = box.cls[0]
            class_name = model.names[int(class_id)]

            # Convert coordinates to integers
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Add label
            label = f'{class_name}: {confidence:.2f}'
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Add FPS to the frame
        cv2.putText(frame, f'FPS: {processing_fps:.1f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('YOLO Detection', frame)

        # Save the frame if output saving is enabled
        if save_output and not isinstance(source, int):
            out.write(frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    if save_output and not isinstance(source, int):
        out.release()
    cv2.destroyAllWindows()

    if not isinstance(source, int):
        print("\nProcessing complete!")
        if save_output:
            print(f"Output saved as: {output_path}")


if __name__ == "__main__":
    # Example usage for webcam
    # run_yolo_detection(0)

    # Example usage for video file
    run_yolo_detection('traffic.mp4')