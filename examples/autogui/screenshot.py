import cv2
import pyautogui
import numpy as np

from xlab.core.resources import get_screeshot_path

# 截取屏幕
screenshot = pyautogui.screenshot(get_screeshot_path("origin"))

# 将图像转换为灰度图像
gray_image = np.array(screenshot).astype(np.uint8)

# 将图像转换为二值图像
binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)[1]

cv2.imwrite(get_screeshot_path("binary"), binary_image)

cv2.imwrite(
    get_screeshot_path("output"), binary_image, [cv2.IMWRITE_PNG_COMPRESSION, 100]
)
