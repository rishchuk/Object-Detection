import tensorflow as tf
import tensorflow_hub as hub
import cv2


model = "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1"
# model = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
# model = "https://www.kaggle.com/models/tensorflow/ssd-mobilenet-v1/TensorFlow2/fpn-640x640/1"
detector = hub.load(model).signatures['default']

video_path = './data/cars.mp4'
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    converted_image = tf.image.convert_image_dtype(frame_rgb, tf.float32)[tf.newaxis, ...]
    result = detector(converted_image)

    boxes = result["detection_boxes"]
    class_ids = result["detection_class_entities"]
    scores = result['detection_scores']

    for box, cls_id, score in zip(boxes, class_ids, scores):
        class_name = str(cls_id.numpy(), 'utf-8')
        if class_name in ['Car', 'Bus', 'Truck', 'Person'] and score > 0.20:
            y1, x1, y2, x2 = box
            print(box)
            cv2.rectangle(frame, (int(x2 * frame_width), int(y1 * frame_height)),
                          (int(x1 * frame_width), int(y2 * frame_height)), (0, 255, 0), 2)
            cv2.putText(frame, f'{class_name}', (int(x1 * frame_width), int(y1 * frame_height) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    out.write(frame)

    cv2.imshow('Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
