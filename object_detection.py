import cv2
import random
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from ultralytics import YOLO
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from datetime import datetime
from database import DB


class KalmanFilter:
    def __init__(self):
        self.dimensions = 2
        self.time_step = 1.0
        self.state_transition = np.eye(2 * self.dimensions)
        self.state_transition[:self.dimensions, self.dimensions:] = self.time_step * np.eye(self.dimensions)
        self.observation_matrix = np.eye(self.dimensions, 2 * self.dimensions)

        self.position_noise_factor = 1.0 / 20
        self.velocity_noise_factor = 1.0 / 160

    def initiate(self, measurement):
        initial_state = np.zeros(4)
        initial_state[:2] = measurement
        std_dev = np.array([
            2 * self.position_noise_factor * measurement[1],
            2 * self.position_noise_factor * measurement[1],
            10 * self.velocity_noise_factor * measurement[1],
            10 * self.velocity_noise_factor * measurement[1]
        ])
        covariance_matrix = np.diag(std_dev ** 2)
        return initial_state, covariance_matrix

    def predict(self, state, covariance):
        predicted_state = self.state_transition @ state
        std_dev = np.array([
            self.position_noise_factor * state[1],
            self.position_noise_factor * state[1],
            self.velocity_noise_factor * state[1],
            self.velocity_noise_factor * state[1]
        ])
        process_covariance = np.diag(std_dev ** 2)
        predicted_covariance = self.state_transition @ covariance @ self.state_transition.T + process_covariance
        return predicted_state, predicted_covariance

    def update(self, state, covariance, measurement):
        projected_state = self.observation_matrix @ state
        projected_covariance = self.observation_matrix @ covariance @ self.observation_matrix.T

        position_noise_std = self.position_noise_factor * state[1]
        measurement_noise_covariance = np.diag([position_noise_std ** 2, position_noise_std ** 2])

        innovation_covariance = projected_covariance + measurement_noise_covariance
        kalman_gain = covariance @ self.observation_matrix.T @ np.linalg.inv(innovation_covariance)

        innovation = measurement - projected_state
        updated_state = state + kalman_gain @ innovation
        updated_covariance = covariance - kalman_gain @ projected_covariance @ kalman_gain.T

        return updated_state, updated_covariance


class ObjectTracker:
    def __init__(self, object_id, center, cls_name, first_visible_frame):
        self.id = object_id
        self.center = center
        self.cls_name = cls_name
        self.first_visible_frame = first_visible_frame
        self.last_seen_frame = first_visible_frame
        self.trajectory = [center]
        self.is_active = True
        self.color = self.generate_unique_color()

        self.kf = KalmanFilter()
        self.state, self.covariance = self.kf.initiate(np.array(center))

    def update(self, new_center, current_frame):
        self.state, self.covariance = self.kf.update(self.state, self.covariance, np.array(new_center))
        self.center = new_center
        self.last_seen_frame = current_frame
        self.trajectory.append(new_center)

    def predict(self):
        self.state, self.covariance = self.kf.predict(self.state, self.covariance)
        return int(self.state[0]), int(self.state[1])

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

    DISTANCE_THRESHOLD = 50

    def __init__(self, model_name, video, output_video, config, flags):
        super().__init__()
        self.db = DB(config)
        self.model = YOLO(model_name)
        self.model_name = model_name.split('/')[-1]
        self.video_name = video.split('/')[-1]
        self.video_id = self.db.register_video(self.video_name)
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

    def create_new_tracker(self, i, current_centers, cls_ids):
        self.object_counter += 1
        new_tracker = ObjectTracker(
            self.object_counter,
            current_centers[i],
            self.model.names[int(cls_ids[i])],
            self.current_frame
        )
        self.object_trackers[self.object_counter] = new_tracker

        self.db.add_trajectory(
            self.model_name,
            self.object_counter,
            self.model.names[int(cls_ids[i])],
            int(current_centers[i][0]),
            int(current_centers[i][1]),
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            self.video_id
        )

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

            current_centers = np.array([self.calculate_center(box) for box in boxes])

            prev_centers = np.array(
                [tracker.predict() for tracker in self.object_trackers.values() if tracker.is_active]
            )

            if len(current_centers) > 0 and len(prev_centers) > 0:
                distances = cdist(current_centers, prev_centers)
                row_indices, col_indices = linear_sum_assignment(distances)

                for i, j in zip(row_indices, col_indices):
                    if distances[i, j] < self.DISTANCE_THRESHOLD:
                        active_tracker_ids = [tid for tid, tracker in self.object_trackers.items() if tracker.is_active]
                        tracker_id = active_tracker_ids[j]

                        self.object_trackers[tracker_id].update(current_centers[i], self.current_frame)
                        self.db.add_trajectory(
                            self.model_name,
                            tracker_id,
                            self.model.names[int(cls_ids[i])],
                            int(current_centers[i][0]),
                            int(current_centers[i][1]),
                            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            self.video_id
                        )
                    else:
                        self.create_new_tracker(i, current_centers, cls_ids)
            elif len(current_centers) > 0:
                for i in range(len(current_centers)):
                    self.create_new_tracker(i, current_centers, cls_ids)

            for tracker_id, tracker in self.object_trackers.items():
                if tracker.is_active:
                    x, y = tracker.predict()
                    if not (0 <= x < self.frame_width and 0 <= y < self.frame_height):
                        tracker.is_active = False
                        continue

                    cv2.putText(frame, f'ID: {tracker_id}', (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    if len(tracker.trajectory) > 1:
                        for i in range(1, len(tracker.trajectory)):
                            cv2.line(frame, tracker.trajectory[i - 1],
                                     tracker.trajectory[i], tracker.color, 2)
                        cv2.circle(frame, tracker.trajectory[-1], 3, tracker.color, -1)

            self.out.write(frame)
            self.update_frame.emit(frame)

        self.cap.release()
        self.out.release()
        self.db.close()

    def stop(self):
        self.running = False
        self.quit()
        self.wait()
