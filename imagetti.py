from ultralytics import YOLO
import cv2
import math

# ----------------------------
# PARAMETERS
# ----------------------------

CAMERA_HEIGHT_M = 1.2
PITCH_DEG = 8.0
SPEED_KMH = 40

# ----------------------------
# DISTANCE ESTIMATION
# ----------------------------

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

    distance_m = (
        CAMERA_HEIGHT_M /
        math.tan(total_angle)
    )

    return distance_m


# ----------------------------
# LOAD MODEL
# ----------------------------

model = YOLO(
    r"runs\detect\training_outputs\suspension_v2\weights\best.pt"
)

# ----------------------------
# INPUT IMAGE
# ----------------------------

image_path = r"pothole.jpg"

image = cv2.imread(image_path)

frame_height, frame_width = image.shape[:2]

FOCAL_PX = int(frame_width * 0.9)

# ----------------------------
# DETECTION
# ----------------------------

results = model.predict(
    source=image,
    conf=0.5,
    verbose=False
)

output = image.copy()

for box in results[0].boxes:

    class_id = int(box.cls[0])

    class_name = model.names[class_id]

    # Skip bumps if you only want potholes
    if class_name.lower() != "pothole":
        continue

    x1, y1, x2, y2 = map(
        int,
        box.xyxy[0]
    )

    distance = estimate_distance_groundplane(
        y2,
        frame_height,
        FOCAL_PX
    )

    if distance is None:
        continue

    speed_mps = SPEED_KMH / 3.6

    tti = distance / speed_mps

    cv2.rectangle(
        output,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    cv2.putText(
        output,
        f"TTI: {tti:.2f} s",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

# ----------------------------
# SAVE RESULT
# ----------------------------

cv2.imwrite(
    "tti_result.jpg",
    output
)

print("Saved: tti_result.jpg")