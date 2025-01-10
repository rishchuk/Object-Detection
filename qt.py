import shutil

from PyQt5.QtGui import QPixmap, QImage

from object_detection import ObjectDetection

import os
import sys
import cv2
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QLabel, QCheckBox, QComboBox, QWidget, QMessageBox, QGridLayout, QTabWidget
)
from PyQt5.QtCore import Qt


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

        download_button = QPushButton("Download processed video")
        download_button.clicked.connect(self.download_processed)
        self.video_layout.addWidget(download_button)

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

        db_config = {
            'host': os.getenv('DB_HOST'),
            'port': int(os.getenv('DB_PORT', 3306)),
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD'),
            'database': os.getenv('DB_DATABASE')
        }

        self.output_video_path = f'output_{self.selected_video.split("/")[-1]}'

        self.detection_instance = ObjectDetection(
            model_name=self.selected_model,
            video=self.selected_video,
            output_video=f'output_{self.selected_video.split("/")[-1]}',
            config=db_config,
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

    def closeEvent(self, event):
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
        event.accept()
