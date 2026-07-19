import time
import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Define table zones (change these coordinates to match your video)
table_zones = {
    "Table 1": (50, 100, 250, 300),
    "Table 2": (50, 100, 250, 300),
}

# Open video
cap = cv2.VideoCapture(
    r"C:\Users\thanh\OneDrive\Documents\GitHub\restaurent-ai\videos\restaurent_video(2).webm"
)

customer_times = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True, classes=[0])

    for result in results:
        boxes = result.boxes

        if boxes.id is None:
            continue

        ids = boxes.id.int().cpu().tolist()

        # Draw table zones and count people
        for table, (tx1, ty1, tx2, ty2) in table_zones.items():

            cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), (255, 0, 0), 2)

            count = 0

            for box in boxes:
                if int(box.cls[0]) == 0:
                    px1, py1, px2, py2 = map(int, box.xyxy[0])

                    cx = (px1 + px2) // 2
                    cy = (py1 + py2) // 2

                    if tx1 < cx < tx2 and ty1 < cy < ty2:
                        count += 1

            cv2.putText(
                frame,
                f"{table}: {count} People",
                (tx1, ty1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

        # Draw person boxes and waiting time
        for box, person_id in zip(boxes, ids):

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if person_id not in customer_times:
                customer_times[person_id] = time.time()

            wait_time = int(time.time() - customer_times[person_id])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.putText(
                frame,
                f"ID:{person_id} Wait:{wait_time}s",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

    cv2.imshow("Restaurant AI - Person Detection", frame)

    key = cv2.waitKey(30) & 0xFF
    if key == ord("q") or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
