from ultralytics import YOLO
import torch
import os

def main():

    # ── Verify GPU availability ────────────────────────────────────────────
    print(f"GPU available: {torch.cuda.is_available()}")
    print(f"GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    device = 0 if torch.cuda.is_available() else "cpu"

    # ── Load pretrained YOLOv8s ───────────────────────────────────────────
    model = YOLO("yolov8n.pt")

    # ── Train ─────────────────────────────────────────────────────────────
    results = model.train(

        # ── DATA ──────────────────────────────────────────────────────
        data=r"pothole-small-1\data.yaml",   # path to your data config
        imgsz=640,

        # ── TRAINING DURATION ─────────────────────────────────────────
        epochs=30,
        patience=20,

        # ── BATCH & COMPUTE ───────────────────────────────────────────
        batch=16,
        device=device,

        # IMPORTANT FIX FOR WINDOWS
        workers=0,  # Set to 0 for Windows to avoid multiprocessing issues and 4 for Linux/Mac

        # ── OPTIMIZER & LEARNING RATE ─────────────────────────────────
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,

        # ── AUGMENTATION ──────────────────────────────────────────────
        augment=False,
        # hsv_h=0.015,
        # hsv_s=0.7,
        # hsv_v=0.4,
        # flipud=0.0,
        # fliplr=0.5,
        # mosaic=1.0,
        # mixup=0.1,
        # copy_paste=0.0,

        # ── OUTPUT & SAVING ───────────────────────────────────────────
        project="training_outputs",
        name="suspension_v1",
        save=True,
        save_period=10,

        # safer on Windows
        plots=False,

        exist_ok=True,

        # ── LOSS WEIGHTS ──────────────────────────────────────────────
        box=7.5,
        cls=0.5,
        dfl=1.5,

        # ── WARMUP ────────────────────────────────────────────────────
        # warmup_epochs=3,
        # warmup_momentum=0.8,
    )

    print("\nTraining complete!")
    print("Best model saved at: training_outputs/suspension_v1/weights/best.pt")


# Required for Windows multiprocessing
if __name__ == "__main__":
    main()