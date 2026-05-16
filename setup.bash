#!/bin/bash

echo "========================================================"
echo "   Edge-based Object Detection Environment Setup (Linux)"
echo "========================================================"
echo ""

echo "[1/6] Checking NVIDIA GPU..."
nvidia-smi
echo ""

echo "[2/6] Creating Virtual Environment (Python 3.11)..."
# In Ubuntu, 'py' isn't used. We use 'python3.11' specifically.
# Note: If this fails, you may need to run: sudo apt install python3.11-venv
python3.11 -m venv suspension_env
echo ""

echo "[3/6] Activating Environment and Verifying Python Version..."
# Linux uses 'source' and a different path to activate the environment
source suspension_env/bin/activate

# Verify it activated correctly and is using 3.11
python --version
python -m pip install --upgrade pip
echo ""

echo "[4/6] Installing PyTorch (Nightly CU128) and all dependencies..."
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
pip install ultralytics roboflow opencv-python matplotlib Pillow pandas numpy scikit-learn tqdm
echo ""

echo "[5/6] Verifying GPU Detection..."
python -c "import torch; print('GPU Available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
echo ""

echo "[6/6] Verifying YOLO Installation..."
python -c "from ultralytics import YOLO; print('YOLOv8 Installation: OK')"
echo ""

echo "========================================================"
echo "   Setup Complete!"
echo "   To work in this project later, run:"
echo "   source suspension_env/bin/activate"
echo "========================================================"