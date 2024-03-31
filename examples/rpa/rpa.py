# 导入所需的库
import cv2
import numpy as np

from xlab.core.resources import get_resource

# 读取图像

img = cv2.imread(get_resource('screenshot.png'))
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 定义颜色的上下界
lower_bound = np.array([0, 0, 0])
upper_bound = np.array([0, 0, 255])

# 创建一个掩膜，指定哪些区域属于上面定义的颜色
mask = cv2.inRange(hsv_img, lower_bound, upper_bound)

# 使用掩膜来提取颜色
output = cv2.bitwise_and(hsv_img, hsv_img, mask=mask)


def get_win_size():
    import ctypes

    user32 = ctypes.windll.user32
    screen_width, screen_height = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    return screen_width, screen_height


def show_images(img1, img2):
    # 确保两个图像有相同的大小
    if img1.shape != img2.shape:
        print('两个图像大小不同，无法比较')
        exit()

    # ### 如果你的图像大小不匹配，你需要做一些预处理来改变他们的大小
    # img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # 水平堆叠图像，你也可以使用np.vstack进行垂直堆叠
    combined = np.hstack((img1, img2))

    # 你的显示窗口尺寸，例如1920x1080，可以根据你实际情况修改
    window_width, window_height = get_win_size()

    # 计算缩放比例并调整图像尺寸
    scale_width = window_width / combined.shape[1]
    scale_height = window_height / combined.shape[0]
    scale = min(scale_width, scale_height)

    # 利用计算的比例进行缩放
    window_size = (int(combined.shape[1] * scale), int(combined.shape[0] * scale))
    resized = cv2.resize(combined, window_size, interpolation=cv2.INTER_CUBIC)

    # 显示缩放后的图像
    cv2.imshow('Combined and Resized', resized)
    cv2.waitKey(0)


if __name__ == "__main__":
    show_images(img, output)
