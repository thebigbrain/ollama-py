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

# 显示图像
# cv2.imshow("Image", binary_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

cv2.imwrite(get_screeshot_path("grey"), gray_image)
cv2.imwrite(get_screeshot_path("output"), binary_image)
