from ultralytics import YOLO
import cv2

# ── LOAD TRAINED MODEL ─────────────────────────────
model = YOLO(
    r"training_outputs\suspension_v1\weights\best.pt"
)

# ── INPUT VIDEO PATH ───────────────────────────────
video_path = r"test.mp4"

# ── OPEN VIDEO ─────────────────────────────────────
cap = cv2.VideoCapture(video_path)

# Check if video opened
if not cap.isOpened():
    print("Error opening video")
    exit()

# ── GET VIDEO PROPERTIES ───────────────────────────
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# ── OUTPUT VIDEO WRITER ────────────────────────────
output_path = "output.mp4"

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out = cv2.VideoWriter(
    output_path,
    fourcc,
    fps,
    (frame_width, frame_height)
)

# ── PROCESS VIDEO FRAME BY FRAME ───────────────────
while True:

    ret, frame = cap.read()

    # End of video
    if not ret:
        break

    # ── RUN YOLO DETECTION ──────────────────────
    results = model.predict(

        source=frame,

        conf=0.5,

        verbose=False
    )

    # ── DRAW DETECTIONS ─────────────────────────
    annotated_frame = results[0].plot()

    # ── SAVE FRAME TO OUTPUT VIDEO ──────────────
    out.write(annotated_frame)

    # ── SHOW OUTPUT ─────────────────────────────
    cv2.imshow("Pothole Detection", annotated_frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ── CLEANUP ────────────────────────────────────────
cap.release()
out.release()

cv2.destroyAllWindows()

print(f"\nSaved output video to: {output_path}")