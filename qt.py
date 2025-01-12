import csv
import random
import shutil
from PyQt5.QtGui import QPixmap, QImage, QBrush, QColor, QPen
from object_detection import ObjectDetection
import os
import cv2
from PyQt5.QtWidgets import (
    QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton,
    QFileDialog, QLabel, QCheckBox, QComboBox, QWidget, QMessageBox, QGridLayout, QTabWidget, QGraphicsEllipseItem,
    QGraphicsScene, QGraphicsView, QLineEdit
)
from PyQt5.QtCore import Qt, QPointF
from database import DB


class DetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Object Detection")
        self.setGeometry(100, 100, 800, 600)
        self.selected_video = None
        self.selected_model = None
        self.selected_flags = []
        self.detection_instance = None
        self.video_thread = None
        self.output_video_path = None
        self.db_config = {
            'host': os.getenv('DB_HOST'),
            'port': int(os.getenv('DB_PORT', 3306)),
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD'),
            'database': os.getenv('DB_DATABASE')
        }
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()

        self.video_label = QLabel("Selected video: None")
        self.video_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.video_label)

        video_button = QPushButton("Select video")
        video_button.clicked.connect(self.select_video)
        main_layout.addWidget(video_button)

        model_layout = QHBoxLayout()
        model_label = QLabel("Select model:")
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "yolov5s.pt",
            "yolov5m.pt",
            "yolov5l.pt",
            "yolov5x.pt",
            "yolov8n.pt",
            "yolov8s.pt",
            "yolov8m.pt",
            "yolov8l.pt",
            "yolov8x.pt",
            "yolov9t.pt",
            "yolov9s.pt",
            "yolov9m.pt",
            "yolov9c.pt",
            "yolov10n.pt",
            "yolov10s.pt",
            "yolov10m.pt",
            "yolov10b.pt",
            "yolov10l.pt",
            "yolov10x.pt",
            "yolo11n.pt",
            "yolo11s.pt",
            "yolo11m.pt",
            "yolo11l.pt",
            "yolo11x.pt",
        ])
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        main_layout.addLayout(model_layout)

        self.flags_layout = QGridLayout()
        self.flag_checkboxes = {
            0: QCheckBox("Person"),
            1: QCheckBox("Bicycle"),
            2: QCheckBox("Car"),
            3: QCheckBox("Motorcycle"),
            4: QCheckBox("Airplane"),
            5: QCheckBox("Bus"),
            6: QCheckBox("Train"),
            7: QCheckBox("Truck"),
            8: QCheckBox("Boat"),
        }

        row, col = 0, 0
        for flag, checkbox in self.flag_checkboxes.items():
            self.flags_layout.addWidget(checkbox, row, col)
            col += 1
            if col == 2:
                col = 0
                row += 1

        main_layout.addLayout(self.flags_layout)

        start_button = QPushButton("Start detection")
        start_button.clicked.connect(self.start_detection)
        main_layout.addWidget(start_button)

        self.tabs = QTabWidget()
        self.video_tab = QWidget()
        self.video_layout = QVBoxLayout()
        self.video_label_display = QLabel("The processed video will be displayed here.")
        self.video_layout.addWidget(self.video_label_display)
        self.video_tab.setLayout(self.video_layout)
        self.tabs.addTab(self.video_tab, "Video output")

        export_button = QPushButton("Export trajectories")
        export_button.clicked.connect(self.export_trajectories)
        self.video_layout.addWidget(export_button)

        download_button = QPushButton("Download processed video")
        download_button.clicked.connect(self.download_processed)
        self.video_layout.addWidget(download_button)

        self.csv_tab = QWidget()
        self.csv_layout = QVBoxLayout()
        self.load_csv_button = QPushButton("Load CSV")
        self.load_csv_button.clicked.connect(self.load_csv)
        self.csv_layout.addWidget(self.load_csv_button)

        self.csv_flags_layout = QGridLayout()
        self.flag_checkboxes = {
            "Person": QCheckBox("Person"),
            "Bicycle": QCheckBox("Bicycle"),
            "Car": QCheckBox("Car"),
            "Motorcycle": QCheckBox("Motorcycle"),
            "Airplane": QCheckBox("Airplane"),
            "Bus": QCheckBox("Bus"),
            "Train": QCheckBox("Train"),
            "Truck": QCheckBox("Truck"),
            "Boat": QCheckBox("Boat"),
        }
        row, col = 0, 0
        for class_name, checkbox in self.flag_checkboxes.items():
            checkbox.stateChanged.connect(self.update_csv_display)
            self.csv_flags_layout.addWidget(checkbox, row, col)
            col += 1
            if col == 5:
                col = 0
                row += 1

        self.csv_layout.addLayout(self.csv_flags_layout)

        self.object_id_input = QLineEdit()
        self.object_id_input.setPlaceholderText("Enter Object ID")
        self.object_id_input.textChanged.connect(self.update_csv_display)
        self.csv_layout.addWidget(self.object_id_input)

        self.csv_display = QGraphicsView()
        self.csv_layout.addWidget(self.csv_display)

        self.csv_tab.setLayout(self.csv_layout)
        self.tabs.addTab(self.csv_tab, "CSV Trajectories")

        main_layout.addWidget(self.tabs)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def select_video(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Select video file", "", "Video files (*.mp4 *.avi)")
        if file_path:
            self.selected_video = file_path
            self.video_label.setText(f"Selected video: {file_path.split('/')[-1]}")

    def start_detection(self):
        if not self.selected_video:
            QMessageBox.warning(self, "Warning", "Please select a video file.")
            return

        self.selected_model = self.model_combo.currentText()
        self.selected_flags = [flag for flag, checkbox in self.flag_checkboxes.items() if checkbox.isChecked()]

        if not self.selected_flags:
            QMessageBox.warning(self, "Warning", "Please select at least one type of object type to track.")
            return

        self.output_video_path = f'output_{self.selected_video.split("/")[-1]}'

        self.detection_instance = ObjectDetection(
            model_name=self.selected_model,
            video=self.selected_video,
            output_video=f'output_{self.selected_video.split("/")[-1]}',
            config=self.db_config,
            flags=self.selected_flags
        )

        self.detection_instance.update_frame.connect(self.update_video_display)
        self.detection_instance.start()

    def update_video_display(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.video_label_display.setPixmap(QPixmap.fromImage(q_image))

    def download_processed(self):
        if not self.output_video_path or not os.path.exists(self.output_video_path):
            QMessageBox.warning(self, "Warning", "Video is not available.")
            return

        save_path, _ = QFileDialog.getSaveFileName(self, "Save video", "", "Video files (*.mp4 *.avi)")
        if save_path:
            try:
                shutil.copy(self.output_video_path, save_path)
                QMessageBox.information(self, "Success", f"Video saved to: {save_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"An error occurred: {e}")

    def export_trajectories(self):
        if not self.selected_video:
            QMessageBox.warning(self, "Warning", "Please select a video first.")
            return

        video_name = self.selected_video.split("/")[-1]

        try:
            db = DB(self.db_config)
            video_id = db.get_video_id(video_name)

            if not video_id:
                QMessageBox.warning(self, "Warning", "No trajectories found for the selected video.")
                return

            trajectories = db.select_trajectories(video_id)
            if not trajectories:
                QMessageBox.information(self, "Info", "No trajectories found to export.")
                return

            save_path, _ = QFileDialog.getSaveFileName(self, "Save trajectories", "", "CSV files (*.csv)")
            if save_path:
                with open(save_path, mode="w", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(["Object ID", "Class", "X", "Y", "Timestamp"])
                    for row in trajectories:
                        writer.writerow(row)

                QMessageBox.information(self, "Success", f"Trajectories exported to: {save_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {e}")

    def load_csv(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Select CSV file", "", "CSV files (*.csv)")
        if file_path:
            try:
                self.trajectories = []
                with open(file_path, mode="r") as file:
                    reader = csv.reader(file)
                    next(reader)
                    for row in reader:
                        self.trajectories.append(row)

                self.update_csv_display()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"An error occurred while loading the CSV: {e}")

    def update_csv_display(self):
        selected_classes = [class_name.lower() for class_name, checkbox in self.flag_checkboxes.items() if checkbox.isChecked()]
        selected_object_id = self.object_id_input.text()

        filtered_trajectories = [
            trajectory for trajectory in self.trajectories
            if (not selected_classes or trajectory[1] in selected_classes) and
               (not selected_object_id or trajectory[0] == selected_object_id)
        ]

        self.visualize_trajectories(filtered_trajectories)

    def visualize_trajectories(self, trajectories):
        scene = QGraphicsScene()
        self.csv_display.setScene(scene)

        colors = {}
        object_points = {}
        for trajectory in trajectories:
            object_id, x, y = trajectory[0], int(trajectory[2]), int(trajectory[3])

            if object_id not in colors:
                colors[object_id] = [random.randint(0, 255) for _ in range(3)]

            if object_id not in object_points:
                object_points[object_id] = []
            object_points[object_id].append(QPointF(x, y))

        for object_id, points in object_points.items():
            if len(points) < 2:
                continue

            for i in range(1, len(points)):
                start_point = points[i - 1]
                end_point = points[i]
                scene.addLine(start_point.x(), start_point.y(), end_point.x(), end_point.y(),
                              QPen(QColor(*colors[object_id]), 2))

            start_point = points[0]
            end_point = points[-1]

            scene.addEllipse(start_point.x() - 3, start_point.y() - 3, 7, 7, QPen(QColor(*colors[object_id])),
                             QBrush(QColor(*colors[object_id])))

            scene.addEllipse(end_point.x() - 3, end_point.y() - 3, 7, 7, QPen(QColor(*colors[object_id])),
                             QBrush(QColor(*colors[object_id])))

        self.csv_display.setScene(scene)

    def closeEvent(self, event):
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
        event.accept()
