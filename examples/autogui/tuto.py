import pyautogui
import subprocess
import time

# Windows系统打开浏览器（这里以谷歌浏览器为例）
subprocess.Popen(["C:/Program Files/Google/Chrome/Application/chrome.exe"])
# 等待浏览器启动
time.sleep(1)

# 输入网址
pyautogui.write("https://www.zhihu.com", interval=0.25)
pyautogui.press("enter")
pyautogui.press("enter")
# 等待网页加载
time.sleep(5)

# 进行其他的操作
# 比如向下滚动页面
pyautogui.scroll(-300)

# 比如点击页面上的某个位置
# 注意这里的坐标需要你根据实际情况调整
pyautogui.click(x=200, y=300)

# 其它的鼠标和键盘操作...
