import sys
import threading


class ThreadGroup:
    _thread_groups = []

    @staticmethod
    def groups():
        return ThreadGroup._thread_groups

    @staticmethod
    def add_daemon(target):
        _t = threading.Thread(target=target, daemon=True)
        ThreadGroup.groups().append(_t)

    @staticmethod
    def start():
        for t in ThreadGroup.groups():
            t.start()

    @staticmethod
    def stop():
        for t in ThreadGroup.groups():
            try:
                print("正在退出 ...", t)
                t.join(timeout=1)
            except Exception as e:
                print("Exception raised:", e)
