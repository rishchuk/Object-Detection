import json
import cv2
import numpy as np
from ultralytics import YOLO
from scipy.spatial.distance import cdist
from datetime import datetime


class ObjectDetection:
    DISTANCE_THRESHOLD = 50

    def __init__(self, model_name, video, output_video):
        self.model_name = YOLO(model_name)
        self.cap = cv2.VideoCapture(video)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), self.fps,
                                   (self.frame_width, self.frame_height))
        self.object_counter = 0
        self.object_tracker = {}
        self.trajectories = {}

    @staticmethod
    def calculate_center(box):
        x1, y1, x2, y2 = box
        return (x1 + x2) // 2, (y1 + y2) // 2

    def run(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            result = self.model_name(source=frame, verbose=False)[0]

            boxes = result.boxes.xyxy.cpu().numpy().squeeze().astype(np.int32)
            cls_ids = result.boxes.cls.cpu().numpy()

            indices = np.where(np.isin(cls_ids, [0, 2, 5, 7]))  # 0: person, 2: car, 5: bus, 7: truck
            boxes = boxes[indices]
            cls_ids = cls_ids[indices]

            current_centers = np.array([self.calculate_center(box) for box in boxes])

            if len(self.object_tracker) == 0:
                for center in current_centers:
                    self.object_counter += 1
                    self.object_tracker[self.object_counter] = center
                    self.trajectories[self.object_counter] = {
                        "class": self.model_name.names[int(cls_ids[0])],
                        "trajectory": [{
                            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            "x": int(center[0]),
                            "y": int(center[1])
                        }]
                    }
            else:
                prev_centers = np.array(list(self.object_tracker.values()))
                distances = cdist(current_centers, prev_centers)

                matched_indices = np.where(distances < self.DISTANCE_THRESHOLD)
                unmatched_objects = set(range(len(current_centers))) - set(matched_indices[0])

                for i, j in zip(matched_indices[0], matched_indices[1]):
                    object_id = list(self.object_tracker.keys())[j]
                    self.object_tracker[object_id] = current_centers[i]

                    self.trajectories[object_id]["trajectory"].append({
                        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        "x": int(current_centers[i][0]),
                        "y": int(current_centers[i][1])
                    })

                for i in unmatched_objects:
                    self.object_counter += 1
                    self.object_tracker[self.object_counter] = current_centers[i]
                    self.trajectories[self.object_counter] = {
                        "class": self.model_name.names[int(cls_ids[0])],
                        "trajectory": [{
                                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                "x": int(current_centers[i][0]),
                                "y": int(current_centers[i][1])
                            }]
                    }

            for object_id, center in self.object_tracker.items():
                x, y = center
                cv2.putText(frame, f'ID: {object_id}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            self.out.write(frame)

            cv2.imshow('Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()

        with open('trajectories.json', 'w') as f:
            json.dump(self.trajectories, f, indent=4)

        print("Trajectories save in 'trajectories.json'")
