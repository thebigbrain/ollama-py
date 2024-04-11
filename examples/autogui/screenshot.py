import cv2
import pyautogui
import numpy as np

from xlab.core.resources import get_screeshot_path


def save_img(path, img):
    gray_image = np.array(img).astype(np.uint8)

    # 将图像转换为二值图像
    binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)[1]

    cv2.imwrite(path, binary_image, [cv2.IMWRITE_PNG_COMPRESSION, 100])


if __name__ == "__main__":
    # 截取屏幕
    screenshot = pyautogui.screenshot(get_screeshot_path("origin"))
    save_img(get_screeshot_path("output"), screenshot)
