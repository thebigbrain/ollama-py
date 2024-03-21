from PyQt6.QtWidgets import (
    QApplication,
)
from xlab.views.tray import ActionConfig, create_tray_icon
from xlab.views.main_win import MainWindow
from xlab.views.chat import ChatFloatingWidget


# 应用程序的主函数
def main():
    app = QApplication([])

    # 创建主窗口和布局
    window = ChatFloatingWidget()

    tray_icon = create_tray_icon(
        parent=app,
        actions=[
            ActionConfig(title="显示", onTrigger=lambda: window.show()),
            ActionConfig(title="退出", onTrigger=app.exit),
        ],
    )

    window.show()

    # 确保在退出时清理系统托盘图标
    app.aboutToQuit.connect(tray_icon.deleteLater)

    # 运行事件循环
    app.exec()


if __name__ == "__main__":
    main()
