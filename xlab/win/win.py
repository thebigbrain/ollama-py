import pygetwindow as gw

from xlab.win.info import get_window_info, print_win_info


def getAllWindowsInfo():
    # 获取所有窗口
    all_windows = gw.getAllWindows()
    _windows_info = []

    for w in all_windows:
        _info = get_window_info(w)
        _windows_info.append(_info)

    return _windows_info


if __name__ == '__main__':
    windows_info = getAllWindowsInfo()
    # 打印所有窗口信息
    for i, info in enumerate(windows_info):
        print(f"Window {i + 1}: ")
        print_win_info(info)
