import pygetwindow as gw
import pyautogui

from xlab.win.info import get_window_info


def get_active_win():
    # 获取当前鼠标的位置
    pos = pyautogui.position()

    # 从所有窗口中找出包含当前鼠标位置的窗口
    active_win = None
    for win in gw.getAllWindows():
        if win.left < pos[0] < win.right and win.top < pos[1] < win.bottom:
            active_win = win
            break

    if active_win is None:
        print("No window found at current mouse position.")

    return active_win


def get_active_win_info():
    return get_window_info(get_active_win())
