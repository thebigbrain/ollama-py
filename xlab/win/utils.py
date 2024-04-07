import win32gui
import win32process


def get_pid(hWnd):
    thread_id, process_id = win32process.GetWindowThreadProcessId(hWnd)
    return process_id


def get_class_name(win):
    class_name = win32gui.GetClassName(win._hWnd)
    return class_name