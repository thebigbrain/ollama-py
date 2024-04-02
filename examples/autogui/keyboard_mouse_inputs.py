from pynput import mouse, keyboard
from datetime import datetime
from db_operations import DBOperations

db_ops = DBOperations('mongodb://localhost:27017/', 'user_input', 'events')
mouse_moving = False
last_mouse_position = (None, None)


# 键盘监听
def on_key_press(key):
    event = {
        'timestamp': datetime.now(),
        'type': 'key_press',
        'key': str(key),
    }
    db_ops.insert_data(event)


def on_key_release(key):
    event = {
        'timestamp': datetime.now(),
        'type': 'key_release',
        'key': str(key),
    }
    db_ops.insert_data(event)


# 鼠标监听
def on_mouse_move(x, y):
    global mouse_moving, last_mouse_position
    if not mouse_moving:
        mouse_moving = True
        last_mouse_position = (x, y)
        event = {
            'timestamp': datetime.now(),
            'type': 'mouse_move_start',
            'x': x,
            'y': y,
        }
        db_ops.insert_data(event)


def on_mouse_click(x, y, button, pressed):
    global mouse_moving, last_mouse_position
    if mouse_moving:
        mouse_moving = False
        event = {
            'timestamp': datetime.now(),
            'type': 'mouse_move_end',
            'x': x,
            'y': y,
        }
        db_ops.insert_data(event)
        last_mouse_position = (None, None)

    event = {
        'timestamp': datetime.now(),
        'type': 'mouse_click',
        'x': x,
        'y': y,
        'button': str(button),
        'pressed': pressed,
    }
    db_ops.insert_data(event)


# 鼠标监听
def on_scroll(x, y, dx, dy):
    event = {
        'timestamp': datetime.now(),
        'type': 'mouse_scroll',
        'x': x,
        'y': y,
        'dx': dx,
        'dy': dy,
    }
    db_ops.insert_data(event)


if __name__ == '__main__':
    print('Starting keyboard and mouse listeners...')

    # 创建并开始键盘监听
    keyboard_listener = keyboard.Listener(on_press=on_key_press, on_release=on_key_release)
    keyboard_listener.start()

    # 创建并开始鼠标监听
    mouse_listener = mouse.Listener(on_move=on_mouse_move, on_click=on_mouse_click, on_scroll=on_scroll)
    mouse_listener.start()

    keyboard_listener.join()
    mouse_listener.join()
