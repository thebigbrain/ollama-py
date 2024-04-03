import sys
from PyQt6.QtWidgets import QSystemTrayIcon, QMenu
from PyQt6.QtGui import QIcon, QAction
from PyQt6.QtCore import QObject
from xlab.core import icons


ICON_PATH = icons.logo


class SystemTrayIcon(QSystemTrayIcon):
    def __init__(self, icon, parent=None):
        super().__init__(icon, parent)
        self.widget = parent
        self.setToolTip(f"AI助理")

        menu = QMenu(parent)
        open_action = QAction("Open", self)
        open_action.triggered.connect(self.show_app)
        menu.addAction(open_action)
        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(sys.exit)
        menu.addAction(quit_action)
        self.setContextMenu(menu)

        self.activated.connect(self.on_tray_activated)

    def show_app(self):
        self.widget.show()
        self.widget.activateWindow()

    def on_tray_activated(self, reason):
        if reason == QSystemTrayIcon.ActivationReason.Trigger.DoubleClick:
            self.show_app()


def create_tray_icon(parent: QObject = None):
    # 创建一个图标对象
    icon = QIcon(ICON_PATH)  # 请确保 icon.png 文件在您的脚本目录中

    # 创建系统托盘图标
    tray_icon = SystemTrayIcon(icon, parent=parent)
    tray_icon.setVisible(True)

    # 显示系统托盘图标
    tray_icon.show()

    return tray_icon
