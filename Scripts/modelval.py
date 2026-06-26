from ultralytics import YOLO

def main():
    model = YOLO("runs/detect/training_outputs/suspension_v1/weights/best.pt")

    metrics = model.val(
        data="Potholes/Bumps-1/data.yaml",
        save_json=True,
        plots=True
    )

    print(metrics)

if __name__ == "__main__":
    main()