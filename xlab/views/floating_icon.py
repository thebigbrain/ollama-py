import os
import sys
from PyQt6.QtWidgets import QLabel, QWidget, QMenu
from PyQt6.QtGui import QPixmap, QBitmap, QPainter
from PyQt6.QtCore import Qt
from PyQt6.QtCore import QRect
from xlab.views.app import AppContext


class FloatingIcon(QWidget):
    def __init__(self, app_context: AppContext):
        super().__init__()
        self.app_context = app_context
        self.initUI()

    def initUI(self):
        self._setupWindow()
        self._loadIcon()

    def _setupWindow(self):
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(64, 64)
        self.setWindowTitle("XLab")

    def _loadIcon(self):
        icon_path = self._getIconPath()
        pixmap = QPixmap(icon_path)
        if pixmap.isNull():
            print("Unable to load icon:", icon_path)
            return

        label = QLabel(self)
        pixmap = self._applyCircleMaskToPixmap(pixmap)
        label.setPixmap(pixmap)
        label.setFixedSize(pixmap.size())

        self.show()

    def _getIconPath(self):
        base_dir = getattr(
            sys, "_MEIPASS", os.path.dirname(os.path.abspath(__package__))
        )
        return os.path.join(base_dir, "assets", "logo.png")

    def _applyCircleMaskToPixmap(self, pixmap):
        mask = QBitmap(pixmap.size())
        mask.fill(Qt.GlobalColor.transparent)
        painter = QPainter(mask)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(Qt.GlobalColor.black)
        painter.drawEllipse(0, 0, pixmap.width(), pixmap.height())
        painter.end()
        pixmap.setMask(mask)
        return pixmap

    # 实现图标的点击和拖动功能
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.m_drag = True
            self.m_dragPosition = (
                event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            )
            event.accept()
            self.setCursor(Qt.CursorShape.OpenHandCursor)

    def mouseMoveEvent(self, event):
        if Qt.MouseButton.LeftButton and self.m_drag:
            self.move(event.globalPosition().toPoint() - self.m_dragPosition)
            event.accept()

    def mouseReleaseEvent(self, event):
        self.m_drag = False
        self.setCursor(Qt.CursorShape.ArrowCursor)

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        action = menu.addAction("退出登录")
        action.triggered.connect(self.handle_menu_item_1)
        menu.exec(event.globalPos())

    def handle_menu_item_1(self):
        self.app_context.exit()


def create_floating_icon(app_context: AppContext):
    return FloatingIcon(app_context)
