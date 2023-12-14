import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# List of video paths
video_paths = ["traffic3.mp4", "traffic.mp4"]
output_paths = ["output_traffic3.mp4", "output_traffic.mp4"]
