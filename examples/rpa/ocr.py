import time
import pyautogui
import pytesseract

# 设置截屏时间间隔
interval = 10

# 开始截屏
while True:
    # 截屏
    image = pyautogui.screenshot()

    # 提取截屏信息
    text = pytesseract.image_to_string(image)

    # 等待下一个截屏
    time.sleep(interval)
