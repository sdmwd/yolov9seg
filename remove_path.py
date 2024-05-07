import sys

# Print original sys.path
print("Original sys.path:", sys.path)

# Prioritize specific path by moving it to the front
conda_path = "n:\\YOLO\\yolov9seg_test\\.conda\\lib\\site-packages"
global_path = "C:\\Users\\Simon\\AppData\\Roaming\\Python\\Python39\\site-packages"

# Make sure the desired path is in the sys.path
if conda_path in sys.path:
    sys.path.insert(0, sys.path.pop(sys.path.index(conda_path)))

# Optionally, remove the global site-packages path if you don't want it
if global_path in sys.path:
    sys.path.remove(global_path)

# Print modified sys.path
print("Modified sys.path:", sys.path)