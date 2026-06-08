from ultralytics import YOLO
import cv2
import math
import csv

CAMERA_HEIGHT_M = 1.2    # CAMERA PARAMETERS
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

    distance_m = (
        CAMERA_HEIGHT_M /
        math.tan(total_angle)
    )

    return round(distance_m, 2)

model = YOLO(
    r"runs\detect\training_outputs\suspension_v2\weights\best.pt"
)

video_path = r"test.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():

    print("Error opening video")
    exit()

frame_width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

frame_height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

fps = int(
    cap.get(cv2.CAP_PROP_FPS)
)

FOCAL_PX = int(frame_width * 0.9)

output_path = "output.mp4"

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out = cv2.VideoWriter(
    output_path,
    fourcc,
    fps,
    (frame_width, frame_height)
)

csv_file = open(
    "suspension_analysis.csv",
    "w",
    newline=""
)

writer = csv.writer(csv_file)

writer.writerow([
    "frame",
    "confidence",
    "distance_m",
    "real_width_m",
    "risk_score",
    "severity",
    "tti_s",
    "urgency"
])

frame_number = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model.predict(
        source=frame,
        conf=0.5,
        verbose=False
    )

    annotated_frame = frame.copy()

    for box in results[0].boxes:

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )

        conf = float(
            box.conf[0]
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

        risk_score = real_width / max(distance, 1)

        if risk_score < 0.08:
            severity = "Low"

        elif risk_score < 0.18:
            severity = "Medium"

        else:
            severity = "High"


        speed_mps = (
            SPEED_KMH / 3.6
        )

        tti = (
            distance /
            speed_mps
        )

        if tti > 3:
            urgency = "Low"

        elif tti > 1.5:
            urgency = "Medium"

        else:
            urgency = "High"



        if urgency == "High" and severity == "High":
            decision = "ADJUST NOW"

        elif urgency == "High" and severity == "Medium":
            decision = "PREPARE"

        elif urgency == "Medium" and severity == "High":
            decision = "PREPARE"

        else:
            decision = "NORMAL"


        cv2.rectangle(
            annotated_frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            annotated_frame,
            f"R:{risk_score:.2f}",
            (x1, y1 - 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2
        )


        cv2.putText(
            annotated_frame,
            f"D:{distance:.1f}m",
            (x1, y1 - 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2
        )

        cv2.putText(
            annotated_frame,
            f"W:{real_width:.2f}m",
            (x1, y1 - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2
        )

        cv2.putText(
            annotated_frame,
            f"S:{severity}",
            (x1, y1 - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2
        )

        cv2.putText(
            annotated_frame,
            f"TTI:{tti:.2f}s {urgency}",
            (x1, y1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2
        )

        cv2.putText(
            annotated_frame,
            decision,
            (x1, y2 + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )


        writer.writerow([
            frame_number,
            round(conf, 4),
            round(distance, 2),
            round(real_width, 2),
            round(risk_score, 3),
            severity,
            round(tti, 2),
            urgency
        ])

    out.write(
        annotated_frame
    )

    cv2.imshow(
        "Smart Suspension Analysis",
        annotated_frame
    )

    frame_number += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()

out.release()

csv_file.close()

cv2.destroyAllWindows()

print(
    f"\nVideo saved: {output_path}"
)

print(
    "CSV saved: suspension_analysis.csv"
)