from ultralytics import YOLO
import cv2
import time


def run_yolo_webcam(model_path='best.pt', conf_threshold=0.5):
    # Load the YOLO model
    model = YOLO(model_path)

    # Initialize webcam
    cap = cv2.VideoCapture(0)  # Use 0 for default webcam

    # Check if webcam is opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    # Set webcam resolution (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Press 'q' to quit")

    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame")
            break

        # Start time to calculate FPS
        start_time = time.time()

        # Run inference on the frame
        results = model(frame, conf=conf_threshold)[0]

        # Calculate FPS
        fps = 1 / (time.time() - start_time)

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
        cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('YOLO Webcam Detection', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_yolo_webcam()