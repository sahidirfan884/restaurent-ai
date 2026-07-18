import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Open webcam (use 0) or replace with your video path
cap = cv2.VideoCapture("C:\Users\thanh\Downloads\stock-footage-turkey-belek-november-the-first-perspective-the-point-of-view-of-the-cctv-camera-in-the (2).webm")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    annotated_frame = results[0].plot()

    cv2.imshow("Restaurant AI - Person Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
