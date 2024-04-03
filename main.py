from PyQt6.QtWidgets import (
    QApplication,
)
from xlab.views.tray import create_tray_icon


# 应用程序的主函数
def main():
    app = QApplication([])

    tray_icon = create_tray_icon()

    # 确保在退出时清理系统托盘图标
    app.aboutToQuit.connect(tray_icon.deleteLater)

    # 运行事件循环
    app.exec()


if __name__ == "__main__":
    main()
