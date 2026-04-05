import cv2

# Load cascades
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
smile_cascade = cv2.CascadeClassifier("haarcascade_smile.xml")

# Start webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frames = cap.read()

    if not ret:
        print("Camera not working")
        break

    # Convert to grayscale
    grey = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(grey, 1.1, 5)

    for (x, y, w, h) in faces:
        # Draw face rectangle
        cv2.rectangle(frames, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Region of Interest
        roi_gray = grey[y:y+h, x:x+w]
        roi_color = frames[y:y+h, x:x+w]

        # Detect eyes
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 10)
        if len(eyes) > 0:
            cv2.putText(frames, "Eyes Detected", (x, y-30),
                        cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 255), 2)

        # Detect smile
        smile = smile_cascade.detectMultiScale(roi_gray, 1.7, 20)
        if len(smile) > 0:
            cv2.putText(frames, "Smiling", (x, y-10),
                        cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 0, 0), 2)

    # Show output
    cv2.imshow("WEB CAM FACE DETECTION", frames)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()