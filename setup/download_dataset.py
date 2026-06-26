from roboflow import Roboflow
rf = Roboflow(api_key="*****************")  #use api key
project = rf.workspace("jampanis-workspace").project("potholesbumps")
version = project.version(1)
dataset = version.download("yolov8")
                