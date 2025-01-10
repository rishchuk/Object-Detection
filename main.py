import sys
from PyQt5.QtWidgets import QApplication
from qt import DetectionApp


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DetectionApp()
    window.show()
    sys.exit(app.exec())
