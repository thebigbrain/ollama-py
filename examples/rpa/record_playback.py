from pynput import mouse, keyboard
import pyautogui
import time

recorded_events = []
start_time = time.time()

listening = True


# 鼠标移动事件的回调函数
def on_mouse_move(x, y):
    global start_time
    timestamp = time.time() - start_time
    recorded_events.append(('mouse_move', timestamp, (x, y)))


# 鼠标点击事件的回调函数
def on_mouse_click(x, y, button, pressed):
    global start_time
    if pressed:
        timestamp = time.time() - start_time
        recorded_events.append(('mouse_click', timestamp, (x, y, button)))


# 鼠标滚动事件的回调函数
def on_mouse_scroll(x, y, dx, dy):
    global start_time
    timestamp = time.time() - start_time
    recorded_events.append(('mouse_scroll', timestamp, (x, y, dx, dy)))


# 键盘按键事件的回调函数
def on_key_press(key):
    global listening
    global start_time
    timestamp = time.time() - start_time
    if key == keyboard.Key.esc:
        # 如果检测到 Esc 键，设置标志使监听器停止
        listening = False
        return False
    elif hasattr(key, 'char') and key.char:
        # 处理可以打印的按键
        recorded_events.append(('key_press', timestamp, key.char))


# 启动监听器
def start_listening():
    with mouse.Listener(
            on_move=on_mouse_move,
            on_click=on_mouse_click,
            on_scroll=on_mouse_scroll) as mouse_listener, \
            keyboard.Listener(on_press=on_key_press) as keyboard_listener:
        print("Start recording, press 'Esc' to stop...")
        while listening:
            time.sleep(0.1)
        mouse_listener.stop()
        keyboard_listener.stop()


# 回放
# 回放
def playback_events(recorded_events):
    print("Playback starting...")
    last_time = 0
    for event in recorded_events:
        event_type, event_time, event_data = event
        sleep_duration = event_time - last_time
        sleep_duration = max(sleep_duration, 0)  # 确保不会是负数
        if sleep_duration > 0:                   # 如果有需要等待的时间，则等待
            time.sleep(sleep_duration)
        last_time = event_time

        # 根据事件类型执行不同的回放操作
        if event_type == 'mouse_move':
            x, y = event_data
            pyautogui.moveTo(x, y, duration=0.0)  # 设置 duration=0.0 为即时移动
        elif event_type == 'mouse_click':
            x, y, button = event_data
            pyautogui.click(x, y)                # 默认鼠标点击不需要动画效果
        elif event_type == 'mouse_scroll':
            x, y, dx, dy = event_data
            for _ in range(abs(dy)):              # 根据滚动的距离进行多次滚动以模拟快速滚轮滚动
                pyautogui.scroll(int(dy/abs(dy)), x, y)  # 保证滚轮滚动方向正确
        elif event_type == 'key_press':
            pyautogui.press(event_data)


# 关闭pyautogui的安全功能，让脚本能运行鼠标移动到屏幕的角落
pyautogui.FAILSAFE = False

# 运行录制过程
start_listening()
print("Events recorded.")

# 回放录制的事件
playback_events(recorded_events)
