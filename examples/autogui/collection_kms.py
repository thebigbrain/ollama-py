import queue
import threading
import time
from PIL import Image
import numpy as np
import pyautogui
from pynput import keyboard, mouse
import cv2

# 定义事件队列
event_queue = queue.Queue(maxsize=500)


# 定义键盘监听函数
def on_press(key):
    event_queue.put({"type": "keyboard", "event": "press", "key": key})


def on_release(key):
    event_queue.put({"type": "keyboard", "event": "release", "key": key})


# 定义鼠标监听函数
def on_move(x, y):
    event_queue.put({"type": "mouse", "event": "move", "x": x, "y": y})


def on_click(x, y, button, pressed):
    event_queue.put(
        {
            "type": "mouse",
            "event": "click",
            "x": x,
            "y": y,
            "button": button,
            "pressed": pressed,
        }
    )


def on_scroll(x, y, dx, dy):
    event_queue.put(
        {"type": "mouse", "event": "scroll", "x": x, "y": y, "dx": dx, "dy": dy}
    )


def get_screeshot():
    img = pyautogui.screenshot()  # 降低截屏分辨率
    _, buffer = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    return img, buffer


# 定义屏幕变化检测函数
def detect_screen_change():
    _, _prev = get_screeshot()

    while True:
        time.sleep(1)

        img, _cur = get_screeshot()

        # 计算两张图像之间的差异
        diff = cv2.absdiff(_prev, _cur)
        change_rate = np.sum(diff) / (diff.size * 255)

        # 变化率大于阈值则触发截屏
        if change_rate > 0.1:
            print("检测到屏幕变化", change_rate)
            event_queue.put({"type": "screenshot", "image": img})

        # 更新上一帧图像
        _prev = _cur


if __name__ == "__main__":
    # 创建键盘监听器
    keyboard_listener = keyboard.Listener(on_press=on_press, on_release=on_release)

    # 创建鼠标监听器
    mouse_listener = mouse.Listener(
        on_move=on_move, on_click=on_click, on_scroll=on_scroll
    )

    # 初始化屏幕变化检测

    # 启动监听器和线程
    keyboard_listener.start()
    mouse_listener.start()

    # 创建屏幕变化检测线程
    screen_change_thread = threading.Thread(target=detect_screen_change, daemon=True)
    screen_change_thread.start()

    # 循环获取事件
    while True:
        event = event_queue.get()
        # print(event)

    # 停止监听器和线程
    keyboard_listener.stop()
    mouse_listener.stop()
    screen_change_thread.join()
