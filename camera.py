from ultralytics import YOLO
import cv2
from datetime import datetime

STREAM_URL = "http://192.168.137.133:8080/video"

# Load your trained model
model = YOLO(
    r"yolov8n.pt"
)

cap = cv2.VideoCapture(STREAM_URL)

if not cap.isOpened():
    print("Could not connect to stream")
    exit()

# Read first frame to get dimensions
ret, frame = cap.read()

if not ret:
    print("Could not read first frame")
    exit()

# Resize dimensions for display and saving
WIDTH = 1280
HEIGHT = 720

# Create output filename with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"smart_suspension_{timestamp}.mp4"

# Video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(
    output_file,
    fourcc,
    20.0,
    (WIDTH, HEIGHT)
)

# Create fullscreen window
cv2.namedWindow(
    "Smart Suspension Detection",
    cv2.WINDOW_NORMAL
)

cv2.setWindowProperty(
    "Smart Suspension Detection",
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)

frame_count = 0

while True:

    ret, frame = cap.read()

    if not ret:
        print("Stream disconnected")
        break

    frame = cv2.resize(frame, (WIDTH, HEIGHT))

    results = model(
        frame,
        conf=0.4,
        verbose=False
    )

    annotated_frame = results[0].plot()

    # Display
    cv2.imshow(
        "Smart Suspension Detection",
        annotated_frame
    )

    # Save frame
    out.write(annotated_frame)

    frame_count += 1

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Saved {frame_count} frames")
print(f"Video saved as: {output_file}")