from ultralytics import YOLO
import torch
import os

def main():

    print(f"GPU available: {torch.cuda.is_available()}")
    print(f"GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    device = 0 if torch.cuda.is_available() else "cpu"

    model = YOLO("runs\detect\training_outputs\suspension_v2\weights\best.pt")

    results = model.train(

        data=r"potholesbumps\data.yaml",
        imgsz=640,

        epochs=10,
        # patience=20,

        batch=16,
        device=device,

        workers=0,  # Set to 0 for Windows to avoid multiprocessing issues and 4 for Linux/Mac

        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,

        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        flipud=0.0,
        fliplr=0.5,
        # mosaic=1.0,
        # mixup=0.1,
        # copy_paste=0.0,

        project="training_outputs",
        name="suspension_v3",
        save=True,
        save_period=10,

        plots=True,
        save_json=True,

        exist_ok=True,

        box=7.5,
        cls=0.5,
        dfl=1.5,

        # warmup_epochs=3,
        # warmup_momentum=0.8,
    )

    print("\nTraining complete!")
    print("Best model saved at: training_outputs/suspension_v3/weights/best.pt")

if __name__ == "__main__":
    main()