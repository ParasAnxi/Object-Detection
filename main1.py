import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

camera_cap = cv2.VideoCapture(0)

while True:
    _, frame = camera_cap.read()

    results = model(frame, stream=True, conf=0.5)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = f"{model.names[cls]} {conf*100:.1f}%"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("YOLOv8 Webcam Detection", frame)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

camera_cap.release()
cv2.destroyAllWindows()
