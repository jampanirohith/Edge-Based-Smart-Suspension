from ultralytics import YOLO
import cv2

model = YOLO(
    r"runs\detect\training_outputs\suspension_v2\weights\best.pt"
)

video_path = r"test.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error opening video")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

output_path = "output1.mp4"

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out = cv2.VideoWriter(
    output_path,
    fourcc,
    fps,
    (frame_width, frame_height)
)

while True:

    ret, frame = cap.read()

    # End of video
    if not ret:
        break

    results = model.predict(

        source=frame,

        conf=0.5,

        verbose=False
    )

    annotated_frame = results[0].plot()

    out.write(annotated_frame)

    cv2.imshow("Pothole Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press Q to quit
        break

cap.release()
out.release()

cv2.destroyAllWindows()

print(f"\nSaved output video to: {output_path}")