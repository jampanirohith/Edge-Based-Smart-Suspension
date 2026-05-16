from roboflow import Roboflow
rf = Roboflow(api_key="xlS3pb3QgzjFDpIbMZlJ")
project = rf.workspace("neww-utfcl").project("speed-bumps-asto8")
version = project.version(1)
dataset = version.download("yolov8")