from ultralytics import YOLO
import cv2

model = YOLO(
    r"runs\detect\training_outputs\suspension_v2\weights\best.pt"
)

image_path = (
    r"bump.jpg"
)

results = model.predict(

    source=image_path,
    conf=0.3,
    show=True,
    save=True,
    verbose=False
)

for result in results:
    boxes = result.boxes

    print(f"\nDetected Objects: {len(boxes)}")

    for box in boxes:

        confidence = float(box.conf[0])

        class_id = int(box.cls[0])

        print(f"Class ID: {class_id}")
        print(f"Confidence: {confidence:.2f}")