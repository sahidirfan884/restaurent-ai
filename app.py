import time
import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n-pose.pt")

# Define table zones (change these coordinates to match your video)
table_zones = {
    "Table 1": (35, 95, 290, 250),
    "Table 2": (320, 70, 540, 200),
}

# Open video
cap = cv2.VideoCapture(
    r"C:\Users\thanh\OneDrive\Documents\GitHub\restaurent-ai\videos\restaurent_video(2).webm"
)

customer_times = {}
seated_customers = {}

def inside_table(x, y, table):
    x1,y1,x2,y2=table
    return x1 <= x <=x2 and y1 <= y <= y2
    
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True, classes=[0])

    for result in results:
        boxes = result.boxes
        keypoints = result.keypoints
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
                2
            )
            if count==0:
                status="FREE"
                color=(0,255,0)
            else:
                status="OCCUPIED"
                color=(0,0,255)
            cv2.putText(
                frame,
                status,
                (tx1,ty2+25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

        # Draw person boxes and waiting time
        for box, person_id in zip(boxes, ids):

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            kp = keypoints.xy[ids.index(person_id)]

            left_hip = kp[11]
            right_hip = kp[12]
            left_knee = kp[13]
            right_knee = kp[14]

            sitting = False

            if left_hip[1] > 0 and left_knee[1] > 0:
                if abs(left_hip[1] - left_knee[1]) < 40:
                    sitting = True

            if right_hip[1] > 0 and right_knee[1] > 0:
                if abs(right_hip[1] - right_knee[1]) < 40:
                    sitting = True
            
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            table_found = None

            for table_name, zone in table_zones.items():
                if inside_table(center_x, center_y, zone):
                    table_found = table_name
                    break

            if table_found and sitting:

                if person_id not in seated_customers:
                    seated_customers[person_id] = {
                        "table": table_found,
                        "start_time": time.time()
                     }

                wait_time = int(time.time() - seated_customers[person_id]["start_time"])
                if wait_time < 120:
                status = "NORMAL"
                color = (0, 255, 0)

            elif wait_time < 300:
                status = "ATTENTION"
                color = (0, 255, 255)

            else:
                status = "DELAYED"
                color = (0, 0, 255)
                
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(
                frame,
                f"{table_found} | {wait_time}s",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,255,0),
                2
                )

                


        
    cv2.imshow("Restaurant AI - Person Detection", frame)

    key = cv2.waitKey(30) & 0xFF
    if key == ord("q") or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
