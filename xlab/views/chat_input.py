from PyQt6.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QPushButton


class ChatInputBox(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Chat Input")
        self.setGeometry(100, 100, 400, 200)  # 设置窗口位置和大小

        layout = QVBoxLayout()

        self.text_edit = QTextEdit()
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.on_send_clicked)

        layout.addWidget(self.text_edit)
        layout.addWidget(self.send_button)

        self.setLayout(layout)

    def on_send_clicked(self):
        # 这里处理发送消息的逻辑, 暂时只打印消息
        message = self.text_edit.toPlainText()
        print("Message:", message)
        self.text_edit.clear()
