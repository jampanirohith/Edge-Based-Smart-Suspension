from roboflow import Roboflow
rf = Roboflow(api_key="xlS3pb3QgzjFDpIbMZlJ")
project = rf.workspace("jampanis-workspace").project("potholes-bumps-hnwyb")
version = project.version(1)
dataset = version.download("yolov8")