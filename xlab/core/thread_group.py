import threading
from typing import Callable


def stop_thread(t: threading.Thread):
    try:
        print("正在退出", t, "...")
        t.join(timeout=1)
    except Exception as e:
        print("Exception raised:", e)


class DaemonThread(threading.Thread):
    _t: threading.Thread = None

    def __init__(
        self,
        group: None = None,
        target: Callable[..., object] | None = None,
        name: str | None = None,
        args: threading.Iterable[threading.Any] = ...,
        kwargs: threading.Mapping[str, threading.Any] | None = None,
    ) -> None:
        super().__init__(group, target, name, args, kwargs, daemon=True)

    def start(self):
        self._t.start()

    def stop(self):
        try:
            print("正在退出", self._t, "...")
            self._t.join(timeout=1)
        except Exception as e:
            print("Exception raised:", e)


class ThreadGroup:
    _daemon_groups = []

    @staticmethod
    def groups():
        return ThreadGroup._daemon_groups

    @staticmethod
    def add_daemon(d):
        ThreadGroup.groups().append(d)

    @staticmethod
    def start():
        for d in ThreadGroup.groups():
            d.start()

    @staticmethod
    def stop():
        for d in ThreadGroup.groups():
            d.stop()
