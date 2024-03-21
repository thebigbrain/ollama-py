from typing import List
from PyQt6.QtWidgets import QSystemTrayIcon, QMenu
from PyQt6.QtGui import QIcon, QAction
from PyQt6.QtCore import QObject
from xlab.core import icons


ICON_PATH = icons.logo


class ActionConfig:
    title: str
    onTrigger = None

    def __init__(self, title: str, onTrigger=None):
        super()

        self.title = title
        self.onTrigger = onTrigger


def create_tray_icon(parent: QObject, actions: List[ActionConfig]):
    # 创建一个图标对象
    icon = QIcon(ICON_PATH)  # 请确保 icon.png 文件在您的脚本目录中

    # 创建系统托盘图标
    tray_icon = QSystemTrayIcon(icon, parent=parent)
    tray_icon.setVisible(True)

    # 创建右键菜单
    menu = QMenu()

    i = 0
    for a in actions:
        # 将动作添加到菜单
        action: QAction = QAction(a.title, parent=menu)
        action.triggered.connect(a.onTrigger)

        menu.addAction(action)
        if i < len(actions) - 1:
            menu.addSeparator()  # 添加一个分隔线

        i = i + 1

    # 将菜单添加到托盘图标
    tray_icon.setContextMenu(menu)

    # 显示系统托盘图标
    tray_icon.show()

    return tray_icon
