from roboflow import Roboflow
rf = Roboflow(api_key="xlS3pb3QgzjFDpIbMZlJ")
project = rf.workspace("pothole-detector-dmdod").project("pothole-clzln")
version = project.version(1)
dataset = version.download("yolov8")