from PyQt6.QtWidgets import QLabel, QWidget, QApplication, QMenu
from PyQt6.QtGui import QPixmap, QBitmap, QPainter
from PyQt6.QtCore import Qt, QPoint
from xlab.core.resources import get_resource
from xlab.views.chat_input import ChatInputBox


class FloatingIcon(QWidget):
    def __init__(self, app: QApplication):
        super().__init__()
        self.app = app
        self.initUI()

    def initUI(self):
        self._setupWindow()
        self._loadIcon()
        self.moveToBottomRight()

    def _setupWindow(self):
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(64, 64)
        self.setWindowTitle("XLab")

    def _loadIcon(self):
        icon_path = get_resource("logo.png")
        pixmap = QPixmap(icon_path)
        if pixmap.isNull():
            print("Unable to load icon:", icon_path)
            return

        label = QLabel(self)
        pixmap = self._applyCircleMaskToPixmap(pixmap)
        label.setPixmap(pixmap)
        label.setFixedSize(pixmap.size())

        self.show()

    def moveToBottomRight(self):
        # 获得屏幕尺寸
        screen_size = self.app.primaryScreen().availableGeometry()
        # 计算窗口在右下角时的新position
        new_x = screen_size.right() - self.width() - 16
        new_y = screen_size.bottom() - self.height() - 32
        # 移动窗口到新位置
        self.move(QPoint(new_x, new_y))

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

    def showChatInput(self):
        # 显示聊天输入框
        self.chat_input_box = ChatInputBox(self)
        self.chat_input_box.show()

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
            # self.showChatInput()
            event.accept()

    def mouseReleaseEvent(self, event):
        self.m_drag = False
        self.setCursor(Qt.CursorShape.ArrowCursor)

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        action = menu.addAction("退出应用")
        action.triggered.connect(self._handle_exit_app)
        menu.exec(event.globalPos())

    def _handle_exit_app(self):
        self.app.exit()


def create_floating_icon(app: QApplication):
    return FloatingIcon(app)
