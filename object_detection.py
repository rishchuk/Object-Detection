import cv2
import random
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from ultralytics import YOLO
from scipy.spatial.distance import cdist
from datetime import datetime
from database import DB


class ObjectTracker:
    def __init__(self, object_id, center, cls_name, first_visible_frame):
        self.id = object_id
        self.center = center
        self.cls_name = cls_name
        self.first_visible_frame = first_visible_frame
        self.last_seen_frame = first_visible_frame
        self.occlusion_count = 0
        self.trajectory = [center]
        self.is_active = True
        self.color = self.generate_unique_color()

    def update(self, new_center, current_frame):
        self.center = new_center
        self.last_seen_frame = current_frame
        self.trajectory.append(new_center)
        self.occlusion_count = 0
        self.is_active = True

    def mark_occluded(self):
        self.occlusion_count += 1
        if self.occlusion_count > 10:
            self.is_active = False

    @staticmethod
    def generate_unique_color():
        while True:
            color = (
                random.randint(50, 200),
                random.randint(50, 200),
                random.randint(50, 200)
            )
            brightness = sum(color) / 3
            if 100 < brightness < 200:
                return color


class ObjectDetection(QThread):
    update_frame = pyqtSignal(np.ndarray)

    DISTANCE_THRESHOLD = 30
    MAX_OCCLUSION_FRAMES = 20
    OVERLAP_THRESHOLD = 0.3

    def __init__(self, model_name, video, output_video, config, flags):
        super().__init__()
        self.db = DB(config)
        self.model = YOLO(model_name)
        self.model_name = model_name.split('/')[-1]
        self.cap = cv2.VideoCapture(video)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), self.fps,
                                   (self.frame_width, self.frame_height))
        self.object_counter = 0
        self.object_trackers = {}
        self.current_frame = 0
        self.flags = flags
        self.running = True

    @staticmethod
    def calculate_center(box):
        x1, y1, x2, y2 = box
        return (x1 + x2) // 2, (y1 + y2) // 2

    @staticmethod
    def calculate_iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

    def detect_occlusions(self, boxes):
        occlusions = {}
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                iou = self.calculate_iou(boxes[i], boxes[j])
                if iou > self.OVERLAP_THRESHOLD:
                    if i not in occlusions:
                        occlusions[i] = []
                    if j not in occlusions:
                        occlusions[j] = []
                    occlusions[i].append(j)
                    occlusions[j].append(i)
        return occlusions

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            self.current_frame += 1

            result = self.model(source=frame, verbose=False)[0]

            boxes = result.boxes.xyxy.cpu().numpy().squeeze().astype(np.int32)
            cls_ids = result.boxes.cls.cpu().numpy()

            indices = np.where(np.isin(cls_ids, self.flags))
            boxes = boxes[indices]
            cls_ids = cls_ids[indices]

            occlusions = self.detect_occlusions(boxes)

            current_centers = np.array([self.calculate_center(box) for box in boxes])

            for tracker_id in list(self.object_trackers.keys()):
                tracker = self.object_trackers[tracker_id]
                if not tracker.is_active:
                    del self.object_trackers[tracker_id]
                    continue
                tracker.mark_occluded()

            if len(current_centers) > 0:
                prev_centers = np.array(
                    [tracker.center for tracker in self.object_trackers.values() if tracker.is_active])

                if len(prev_centers) > 0:
                    distances = cdist(current_centers, prev_centers)
                    matched_indices = np.where(distances < self.DISTANCE_THRESHOLD)

                    for i, j in zip(matched_indices[0], matched_indices[1]):
                        active_tracker_ids = [tid for tid, tracker in self.object_trackers.items() if tracker.is_active]
                        tracker_id = active_tracker_ids[j]

                        if i in occlusions:
                            print(f"Detected occlusion for object {tracker_id}")

                        self.object_trackers[tracker_id].update(current_centers[i], self.current_frame)
                        self.db.add_trajectory(
                            self.model_name,
                            tracker_id,
                            self.model.names[int(cls_ids[i])],
                            int(current_centers[i][0]),
                            int(current_centers[i][1]),
                            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        )

                for i, center in enumerate(current_centers):
                    if not any(np.linalg.norm(np.array(center) - np.array(prev_center)) < self.DISTANCE_THRESHOLD
                               for prev_center in prev_centers):
                        self.object_counter += 1
                        new_tracker = ObjectTracker(
                            self.object_counter,
                            center,
                            self.model.names[int(cls_ids[i])],
                            self.current_frame
                        )
                        self.object_trackers[self.object_counter] = new_tracker

                        self.db.add_trajectory(
                            self.model_name,
                            self.object_counter,
                            self.model.names[int(cls_ids[i])],
                            int(center[0]),
                            int(center[1]),
                            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        )

            for tracker_id, tracker in self.object_trackers.items():
                if tracker.is_active:
                    x, y = tracker.center
                    cv2.putText(frame, f'ID: {tracker_id}', (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    if len(tracker.trajectory) > 1:
                        for i in range(1, len(tracker.trajectory)):
                            cv2.line(
                                frame, tracker.trajectory[i - 1],
                                tracker.trajectory[i], tracker.color, 2
                            )

                        cv2.circle(
                            frame, tracker.trajectory[-1], 3, tracker.color, -1
                        )

            self.out.write(frame)
            self.update_frame.emit(frame)

        self.cap.release()
        self.out.release()
        self.db.close()

    def stop(self):
        self.running = False
        self.quit()
        self.wait()
