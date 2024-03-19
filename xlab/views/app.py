import sys
from PyQt6.QtWidgets import QApplication


class AppContext:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._app = None
        return cls._instance

    @property
    def application(self):
        if self._app is None:
            self._app = QApplication(sys.argv)
        return self._app

    def start(self):
        if self._app is not None:
            sys.exit(self._app.exec())

    def exit(self):
        if self._app is not None:
            self._app.quit()

    @classmethod
    def ensure_single_instance(cls) -> "AppContext":
        """确保全局只有一个AppContext实例"""
        cls()
        _ = cls._instance.application
        return cls._instance


def create_app_context():
    # 这里可以根据需要决定是否返回单例的AppContext
    # 如果需要单例，可以修改ensure_single_instance方法
    return AppContext()


def get_application_context():
    return AppContext.ensure_single_instance()
