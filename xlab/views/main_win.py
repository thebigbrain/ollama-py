from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLineEdit,
    QApplication,
    QPushButton,
)
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import Qt
from xlab.core import icons


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint)
        # 设置窗口大小
        self.setFixedSize(300, 80)  # 假设窗口大小为300x200像素
        self.move_to_bottom_right()

        # 创建一个垂直布局管理器
        layout = QVBoxLayout(self)

        # 创建输入框
        self.line_edit = QLineEdit(self)
        self.line_edit.setPlaceholderText("请输入内容...")

        # 创建最小化按钮
        self.button_minimize = QPushButton("最小化", self)
        self.button_minimize.clicked.connect(self.hide)

        # 添加输入框和按钮到布局
        layout.addWidget(self.line_edit)
        layout.addWidget(self.button_minimize)
        self.setLayout(layout)

        # 设置窗口标题和图标
        self.setWindowTitle("系统托盘应用")
        self.setWindowIcon(QIcon(icons.logo))

    def hide(self):
        """重写隐藏行为，将窗口最小化到系统托盘"""
        self.setVisible(False)

    def show_normal(self):
        """重写的 show 方法，用于从系统托盘中恢复窗口"""
        self.setVisible(True)
        self.setWindowState(self.windowState() & ~Qt.WindowState.WindowMinimized)
        self.activateWindow()

    def move_to_bottom_right(self):
        # 获取屏幕尺寸
        screen_geom = QApplication.primaryScreen().geometry()

        # 计算窗口偏移位置
        xpos = screen_geom.width() - self.width() - 15
        ypos = screen_geom.height() - self.height() - 50

        # 移动窗口到右下角
        self.move(xpos, ypos)
