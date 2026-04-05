import cv2
from ultralytics import YOLO

# Load YOLO pose model
model = YOLO('yolov8n-pose.pt')

# Give correct video path
video_path = r"WHAT_A_RACE_Men_s_100m_Final_Paris2024_highlights_1080P.mp4"

# Open video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Cannot open video")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Video finished")
        break

    # Run pose detection
    results = model.predict(frame, verbose=False)

    # Draw pose keypoints
    annotated_frame = results[0].plot()

    # Show output
    cv2.imshow("Pose Detection", annotated_frame)

    # Press q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()