from xlab.win.utils import get_class_name, get_pid


def get_window_info(win):
    _info = {
        'class_name': get_class_name(win),
        'process_id': get_pid(win._hWnd),
        'title': win.title,  # 窗口标题
        'size': win.size,  # 窗口大小（宽，高）
        'top_left': win.topleft,  # 窗口左上角的坐标（x，y）
        'box': win.box,  # 左，上，宽，高四个属性的元组
        'is_active': win.isActive,  # 窗口是否在前台
        'is_maximized': win.isMaximized,  # 窗口是否最大化
        'is_minimized': win.isMinimized  # 窗口是否最小化
    }
    return _info

def print_win_info(info):
    print(f"\tClassName: {info['class_name']}")
    print(f"\tProcess ID: {info['process_id']}")
    print(f"\tTitle: {info['title']}")
    print(f"\tSize: {info['size']}")
    print(f"\tTop-Left Corner: {info['top_left']}")
    print(f"\tBox: {info['box']}")
    print(f"\tIs Active: {info['is_active']}")
    print(f"\tIs Maximized: {info['is_maximized']}")
    print(f"\tIs Minimized: {info['is_minimized']}")