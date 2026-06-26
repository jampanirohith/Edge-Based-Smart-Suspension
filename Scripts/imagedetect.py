from ultralytics import YOLO
import cv2
import math

CAMERA_HEIGHT_M = 1.2
PITCH_DEG = 8.0
SPEED_KMH = 40

def estimate_distance_groundplane(y_bottom, image_height, focal_px):

    pitch_rad = math.radians(PITCH_DEG)

    cy = image_height / 2

    pixel_offset = y_bottom - cy

    angle_rad = math.atan(pixel_offset / focal_px)

    total_angle = pitch_rad + angle_rad

    if total_angle <= 0:
        return None

    distance = CAMERA_HEIGHT_M / math.tan(total_angle)

    return distance

model = YOLO(
    r"runs\detect\training_outputs\suspension_v2\weights\best.pt"
)

image_path = r"Resources\bumps.jpg"

frame = cv2.imread(image_path)

frame_height, frame_width = frame.shape[:2]

FOCAL_PX = int(frame_width * 0.9)

results = model.predict(
    source=frame,
    conf=0.5,
    verbose=False
)

output = frame.copy()

speed_mps = SPEED_KMH / 3.6

for box in results[0].boxes:

    x1, y1, x2, y2 = map(int, box.xyxy[0])

    confidence = float(box.conf[0])

    bbox_width = x2 - x1

    distance = estimate_distance_groundplane(
        y2,
        frame_height,
        FOCAL_PX
    )

    if distance is None:
        continue

    real_width = (bbox_width * distance) / FOCAL_PX

    risk_score = real_width / max(distance, 1)

    if risk_score < 0.08:
        severity = "Low"

    elif risk_score < 0.18:
        severity = "Medium"

    else:
        severity = "High"


    tti = distance / speed_mps

    if tti > 3:
        urgency = "Low"

    elif tti > 1.5:
        urgency = "Medium"

    else:
        urgency = "High"


    if urgency == "High" and severity == "High":
        decision = "ADJUST NOW"

    elif (urgency == "High" and severity == "Medium") or \
         (urgency == "Medium" and severity == "High"):
        decision = "PREPARE"

    else:
        decision = "NORMAL"


    cv2.rectangle(
        output,
        (x1, y1),
        (x2, y2),
        (0,255,0),
        3
    )

    cv2.putText(
        output,
        f"Confidence : {confidence:.2f}",
        (x1, y1-110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,0),
        2
    )

    cv2.putText(
        output,
        f"Distance : {distance:.2f} m",
        (x1, y1-80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0,255,0),
        2
    )

    cv2.putText(
        output,
        f"Severity : {severity}",
        (x1, y1-50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0,255,255),
        2
    )

    cv2.putText(
        output,
        f"TTI : {tti:.2f} s",
        (x1, y1-20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,255),
        2
    )

    if decision == "ADJUST NOW":
        color = (0,0,255)

    elif decision == "PREPARE":
        color = (0,165,255)

    else:
        color = (0,255,0)

    cv2.putText(
        output,
        decision,
        (x1, y2+35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        3
    )


output_path = "Resources\\imageresult.png"

cv2.imwrite(output_path, output)

print("Saved :", output_path)

cv2.imshow("Result", output)
cv2.waitKey(0)
cv2.destroyAllWindows()