import sys
from PyQt6.QtWidgets import QApplication, QSystemTrayIcon, QMenu, QAction
from PyQt6.QtGui import QIcon


def create_tray_icon(app: QApplication):
    # 创建一个图标对象
    icon = QIcon("assets/logo.jpeg")  # 请确保 icon.png 文件在您的脚本目录中

    # 创建系统托盘图标
    tray_icon = QSystemTrayIcon(app)
    tray_icon.setIcon(icon)
    tray_icon.setVisible(True)

    # 创建右键菜单
    menu = QMenu()

    # 创建退出动作
    exit_action = QAction("退出", menu)
    exit_action.triggered.connect(app.exit)  # 退出动作连接到app的退出方法
    menu.addAction(exit_action)

    # 将菜单添加到托盘图标
    tray_icon.setContextMenu(menu)

    # 显示系统托盘图标
    tray_icon.show()
