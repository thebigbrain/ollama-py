from typing import Sequence
import ollama
from PyQt6.QtCore import QThread, pyqtSignal


class MessageThread(QThread):
    # 创建一个信号，当消息准备发送时发射
    messageReady = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, message_stream, message):
        super().__init__()
        self.message_stream = message_stream
        self.message = message

    def run(self):
        # 这个方法在新线程中运行
        self.message_stream.add_message(self.message)
        for chunk in self.message_stream.send():
            content = chunk["message"]["content"]
            self.messageReady.emit(content)  # 在这里发射信号

            if chunk["done"]:
                self.finished.emit()


class MessageStream:
    def __init__(self):
        self.messages: Sequence[ollama.Message] = []

    def add_message(self, message):
        """添加用户消息到消息流中"""
        if message:  # 确保非空消息
            self.messages.append({"role": "user", "content": message})

        return self

    def add_system_message(self, message):
        """添加系统消息到消息流中"""
        if message:  # 确保非空消息
            self.messages.append({"role": "system", "content": message})
        return self

    def add_assistant_message(self, message):
        if message:  # 确保非空消息
            self.messages.append({"role": "assistant", "content": message})
        return self

    def send(self):
        """获取消息流"""
        return ollama.chat(
            model="codellama",
            messages=self.messages,
            stream=True,
        )
