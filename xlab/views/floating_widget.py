from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QPushButton, QWidget


class FloatingWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Floating Widget")
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.Tool)

        btn = QPushButton("Click Me", self)
        btn.clicked.connect(self.on_click)
        btn.resize(btn.sizeHint())
        self.resize(btn.sizeHint())

    def on_click(self):
        print("Button has been clicked")


if __name__ == "__main__":
    app = QApplication([])

    floating_widget = FloatingWidget()
    floating_widget.show()

    app.exec()
