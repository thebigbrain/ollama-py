from PyQt6.QtWidgets import (
    QApplication,
    QTextEdit,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeyEvent
from xlab.chat.message import MessageStream


class MyLineEdit(QLineEdit):
    def __init__(self, parent=None):
        super(MyLineEdit, self).__init__(parent)

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_Return or event.key() == Qt.Key.Key_Enter:
            if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                # Ctrl+Enter was pressed - insert a new line
                self.insert("\n")
            else:
                # Enter was pressed - emit the returnPressed signal
                self.returnPressed.emit()
        else:
            super().keyPressEvent(event)  # Propagate other key events to the base class


class ChatFloatingWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.stream = MessageStream()

    def initUI(self):
        # 设置窗口标题和大小
        self.setWindowTitle("AI助理")
        self.setFixedSize(300, 400)
        self.setWindowFlags(
            self.windowFlags() | Qt.WindowType.Tool | Qt.WindowType.WindowStaysOnTopHint
        )

        # 创建布局
        layout = QVBoxLayout()

        # 聊天历史区域
        self.chatHistory = QTextEdit()
        self.chatHistory.setReadOnly(True)
        layout.addWidget(self.chatHistory)

        # 文本输入框
        self.messageInput = MyLineEdit(self)
        self.messageInput.setPlaceholderText("按回车发送，按Ctrl + 回车换行")
        self.messageInput.returnPressed.connect(lambda: self.sendMessage())
        layout.addWidget(self.messageInput)

        # 应用布局
        self.setLayout(layout)
        self.move_to_bottom_right()

    def sendMessage(self):
        # 获取输入框内容
        message = self.messageInput.text()
        if message:
            self.messageInput.setText("")
            self.chatHistory.append("You: " + message)
            self.stream.add_message(message=message)
            for chunk in self.stream.send():
                print(chunk["message"]["content"], end="", flush=True)

    def move_to_bottom_right(self):
        # 获取屏幕尺寸
        screen_geom = QApplication.primaryScreen().geometry()

        # 计算窗口偏移位置
        xpos = screen_geom.width() - self.width() - 15
        ypos = screen_geom.height() - self.height() - 80

        # 移动窗口到右下角
        self.move(xpos, ypos)


if __name__ == "__main__":
    app = QApplication([])

    chat_widget = ChatFloatingWidget()
    chat_widget.show()

    app.exec()
