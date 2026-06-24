from ultralytics import YOLO

model = YOLO(
    r"runs\detect\training_outputs\suspension_v2\weights\best.pt"
)

metrics = model.val(
    save_json=True,
    plots=True
)