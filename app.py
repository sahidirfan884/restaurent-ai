import time
import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Open webcam (use 0) or replace with your video path
cap = cv2.VideoCapture(r"C:\Users\thanh\OneDrive\Documents\GitHub\restaurent-ai\videos\restaurent_video(1).webm")

customer_times={}
while True:
    ret, frame = cap.read()
    if not ret:
        break
    results=model.track(frame,persist=True, classes=[0])
    for result in results:
    boxes = result.boxes

    if boxes.id is None:
        continue

    ids = boxes.id.int().cpu().tolist()

    for box, person_id in zip(boxes, ids):
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if person_id not in customer_times:
            customer_times[person_id] = time.time()

        wait_time = int(time.time() - customer_times[person_id])

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

        cv2.putText(
            frame,
            f"ID:{person_id} Wait:{wait_time}s",
            (x1, y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,255,0),
            2
        )

    annotated_frame = results[0].plot()

    cv2.imshow("Restaurant AI - Person Detection", annotated_frame)

    key=cv2.waitKey(30) & 0xFF
    if key==ord('q') or key ==27:
        break

cap.release()
cv2.destroyAllWindows()
