from typing import Sequence
import ollama


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
