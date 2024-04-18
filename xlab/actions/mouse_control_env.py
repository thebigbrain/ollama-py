import numpy as np
import gymnasium as gym


class MouseControlEnv(gym.Env):
    def __init__(self, screen_width, screen_height):
        super(MouseControlEnv, self).__init__()

        # 定义动作空间和状态空间
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(2,), dtype=np.float32
        )  # x和y的移动
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(screen_height, screen_width, 3), dtype=np.uint8
        )  # RGB屏幕

        # 定义其他环境相关参数
        self.target_position = np.array([screen_width / 2, screen_height / 2])
        self.mouse_position = np.array([screen_width / 2, screen_height / 2])

    def step(self, action):
        # 执行动作，更新环境状态
        # 你需要将action映射到实际屏幕坐标

        # 这里我们模拟了这个过程
        self.mouse_position += action * np.array(
            [10, 10]
        )  # 假设action[-1, 1]映射到[-10, 10]的屏幕移动

        # 计算奖励，越接近目标，奖励越高
        distance = np.linalg.norm(self.target_position - self.mouse_position)
        reward = -distance  # 使用负距离作为奖励，鼓励靠近目标

        # 检查是否完成任务，这里我们简化处理
        done = distance < 5  # 当鼠标距离目标位置小于5个像素时任务完成

        # 获取下一个状态
        next_state = self.get_screen()

        return next_state, reward, done, {}

    def reset(self):
        # 重置环境到初始状态
        self.mouse_position = np.array(
            [self.observation_space.shape[1] / 2, self.observation_space.shape[0] / 2]
        )
        return self.get_screen()

    def get_screen(self):
        # 获取当前环境状态的方法，这里需要你根据实际情况实现
        # 可以是屏幕的截图或其他特征表示
        # 这里我们返回了一个全0的屏幕，你需要替换为实际的屏幕获取方式
        return np.zeros(
            (self.observation_space.shape[0], self.observation_space.shape[1], 3),
            dtype=np.uint8,
        )


if __name__ == "__main__":
    # 实例化并使用环境
    env = MouseControlEnv(screen_width=800, screen_height=600)
