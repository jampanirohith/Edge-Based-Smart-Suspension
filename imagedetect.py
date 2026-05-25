from ultralytics import YOLO
import cv2

# ── LOAD TRAINED MODEL ─────────────────────────────────
model = YOLO(
    r"training_outputs\suspension_v1\weights\best.pt"
)

# ── IMAGE PATH ─────────────────────────────────────────
image_path = (
    r"pothole-small-1\train\images\11_jpg.rf.4f03b935bde2988925f5892bab2fc441.jpg"
)

# ── RUN PREDICTION ─────────────────────────────────────
results = model.predict(

    source=image_path,

    conf=0.3,

    show=True,

    save=True,

    verbose=False
)

# ── PRINT DETECTIONS ───────────────────────────────────
for result in results:

    boxes = result.boxes

    print(f"\nDetected Objects: {len(boxes)}")

    for box in boxes:

        confidence = float(box.conf[0])

        class_id = int(box.cls[0])

        print(f"Class ID: {class_id}")
        print(f"Confidence: {confidence:.2f}")