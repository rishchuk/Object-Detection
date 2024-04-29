from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO('yolov9e-seg.pt')

video_path = './data/cars.mp4'
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

file = video_path.split('/')[-1]

out = cv2.VideoWriter(f'output_{file}', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    result = model(source=frame, verbose=False)[0]

    boxes = result.boxes.xyxy.cpu().numpy().squeeze().astype(np.int32)
    cls_ids = result.boxes.cls

    indices = np.where(np.isin(cls_ids, [0, 2, 5, 7]))
    boxes = boxes[indices]
    cls_ids = cls_ids[indices]

    for box, cls_id in zip(boxes, cls_ids):
        class_name = model.names[int(cls_id)]
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{class_name}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    out.write(frame)

    cv2.imshow('Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
