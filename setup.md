# Edge-based-Object-Detection-using-5G-video-streaming

# Set-up on Windows
```bash
./setup.bat
```

# Set-up on Linux
```bash
chmod +x setup.sh
./setup.sh
```

## 1. Test NVIDIA GPU

Check whether the NVIDIA GPU and drivers are installed correctly.

```bash
nvidia-smi
```

Expected: GPU information table should appear.

---

# 2. Setting Up Environment and Prerequisites

## 2.1 Install Python 3.11

Install Python 3.11:

```bash
py install 3.11
```

Verify installation:

```bash
py -3.11 --version
```

Expected output:

```bash
Python 3.11.x
```

---

## 2.2 Create Virtual Environment

Create a virtual environment inside the project directory:

```bash
py -3.11 -m venv suspension_env
```

This creates an isolated Python environment for the project.

---

## 2.3 Activate Virtual Environment

### Windows PowerShell

```powershell
.\suspension_env\Scripts\Activate
```

Expected output in terminal:

```bash
(suspension_env)
```

---

# 3. Install Required Libraries

## 3.1 Upgrade pip

```bash
python -m pip install --upgrade pip
```

---

## 3.2 Install GPU-enabled PyTorch (RTX 5050)

Install the latest nightly CUDA build for RTX 50-series GPUs:

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

> NOTE:
> RTX 5050 Laptop GPU requires newer CUDA support.
> Older CUDA builds such as cu118 may not work properly.

---

## 3.3 Verify GPU Detection

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected output:

```bash
True NVIDIA GeForce RTX 5050 Laptop GPU
```

---

## 3.4 Install Ultralytics (YOLOv8)

```bash
pip install ultralytics
```

---

## 3.5 Verify YOLO Installation

```bash
python -c "from ultralytics import YOLO; print('YOLOv8 OK')"
```

Expected output:

```bash
YOLOv8 OK
```

---

## 3.6 Install Additional Libraries

### Roboflow (dataset downloading)

```bash
pip install roboflow
```

### Image processing and visualization

```bash
pip install opencv-python matplotlib Pillow
```

### Data processing libraries

```bash
pip install pandas numpy scikit-learn tqdm
```

---

# 4. Final GPU Verification

Run Python:

```bash
python Material\gputest.py
```

Expected output:

```bash
True
NVIDIA GeForce RTX 5050 Laptop GPU
```

---

# 5. Test YOLO on GPU


```bash
python Material\YOLO\yolotest.py
```

Expected output:

```bash
CUDA:0 (NVIDIA GeForce RTX 5050 Laptop GPU)
```

---

# 6. Reactivating Environment Later

Whenever reopening the project, activate the environment again:

```powershell
.\suspension_env\Scripts\Activate
```

Deactivate environment when finished:

```bash
deactivate
```