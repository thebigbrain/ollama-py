import time
import pyautogui
from pynput import mouse, keyboard
from datetime import datetime
from db_operations import DBOperations
from xlab.core.resources import get_screeshot_path


db_ops = DBOperations("mongodb://localhost:27017/", "user_input", "events")


def save_screeshot():
    screenshot = get_screeshot_path(
        "_".join(
            [
                datetime.now().strftime("%Y-%m-%d").replace("-", "_"),
                str(time.time()).replace(".", "_"),
            ],
        )
    )

    pyautogui.screenshot(screenshot)
    return screenshot


def save_event(event, screenshot=None):
    timestamp = datetime.now()

    data = dict({"timestamp": timestamp, "screenshot": screenshot})
    data.update(event)
    db_ops.insert_data(data)


# 键盘监听
def on_key_press(key):
    save_event(
        {
            "type": "key_press",
            "key": str(key),
        }
    )


def on_key_release(key):
    save_event(
        {
            "type": "key_release",
            "key": str(key),
        }
    )


# 鼠标监听
def on_mouse_move(x, y):
    save_event(
        {
            "type": "mouse_move",
            "x": x,
            "y": y,
        }
    )


def on_mouse_click(x, y, button, pressed):
    save_event(
        {
            "type": "mouse_click",
            "x": x,
            "y": y,
            "button": str(button),
            "pressed": pressed,
        }
    )


# 鼠标监听
def on_scroll(x, y, dx, dy):
    save_event(
        {
            "type": "mouse_scroll",
            "x": x,
            "y": y,
            "dx": dx,
            "dy": dy,
        }
    )


if __name__ == "__main__":
    print("Starting keyboard and mouse listeners...")

    # 创建并开始键盘监听
    keyboard_listener = keyboard.Listener(
        on_press=on_key_press, on_release=on_key_release
    )
    keyboard_listener.start()

    # 创建并开始鼠标监听
    mouse_listener = mouse.Listener(
        on_move=on_mouse_move, on_click=on_mouse_click, on_scroll=on_scroll
    )
    mouse_listener.start()

    keyboard_listener.join()
    mouse_listener.join()
