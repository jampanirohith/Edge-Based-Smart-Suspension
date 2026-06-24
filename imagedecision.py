from ultralytics import YOLO
import cv2
import math

CAMERA_HEIGHT_M = 1.2
PITCH_DEG = 8.0
SPEED_KMH = 40

def estimate_distance_groundplane(
    y_bottom,
    image_height,
    focal_px
):

    pitch_rad = math.radians(PITCH_DEG)

    cy = image_height / 2

    pixel_offset = y_bottom - cy

    angle_rad = math.atan(
        pixel_offset / focal_px
    )

    total_angle = pitch_rad + angle_rad

    if total_angle <= 0:
        return None

    return CAMERA_HEIGHT_M / math.tan(total_angle)


model = YOLO(
    r"runs\detect\training_outputs\suspension_v2\weights\best.pt"
)

image = cv2.imread("pothole.jpg")

frame_height, frame_width = image.shape[:2]

FOCAL_PX = int(frame_width * 0.9)

results = model.predict(
    source=image,
    conf=0.5,
    verbose=False
)

output = image.copy()

for box in results[0].boxes:

    cls_id = int(box.cls[0])

    if model.names[cls_id].lower() != "pothole":
        continue

    x1, y1, x2, y2 = map(
        int,
        box.xyxy[0]
    )

    bbox_width = x2 - x1

    distance = estimate_distance_groundplane(
        y2,
        frame_height,
        FOCAL_PX
    )

    if distance is None:
        continue

    real_width = (
        bbox_width *
        distance
    ) / FOCAL_PX

    speed_mps = SPEED_KMH / 3.6

    tti = distance / speed_mps

    if real_width < 0.5:
        severity = "Low"

    elif real_width < 1.0:
        severity = "Medium"

    else:
        severity = "High"

    if tti > 3:
        urgency = "Low"

    elif tti > 1.5:
        urgency = "Medium"

    else:
        urgency = "High"

    if severity == "High" and urgency == "High":
        decision = "ADJUST NOW"

    elif severity == "High" or urgency == "High":
        decision = "PREPARE"

    else:
        decision = "NORMAL"

    cv2.rectangle(
        output,
        (x1,y1),
        (x2,y2),
        (0,255,0),
        2
    )

    cv2.putText(
        output,
        decision,
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0,0,255),
        3
    )

cv2.imwrite(
    "./decision_example.jpg",
    output
)

print("Saved: ./decision_example.jpg")