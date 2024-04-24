import gymnasium
from gymnasium import spaces
import numpy as np


class MouseEnv(gymnasium.Env):
    """一个简化的鼠标环境，AI的任务是移动鼠标并点击目标。"""

    def __init__(self, screen_width, screen_height):
        super(MouseEnv, self).__init__()

        # 定义状态和动作空间
        self.action_space = spaces.Box(
            low=np.array([0, 0, -1]),  # x移动, y移动, 点击(-1表示不点击, 1表示点击)
            high=np.array([screen_width, screen_height, 1]),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(screen_height, screen_width, 3), dtype=np.uint8
        )

        self.state = None  # 当前的状态
        self._target = None
        self.screen_width = screen_width
        self.screen_height = screen_height

    def reset(self):
        """重置环境到初始状态。"""
        self._target = self._generate_target()
        self.state = np.array([self.screen_width / 2.0, self.screen_height / 2.0])
        return self.state

    def step(self, action):
        """执行一个动作并返回新状态，奖励，是否结束，额外信息。"""
        mouse_position = self.state + action[:2]
        # 确保鼠标不能移出屏幕
        mouse_position = np.clip(
            mouse_position, [0, 0], [self.screen_width, self.screen_height]
        )

        # 因为这里只是一个简化例子，并不涉及真实点击操作，所以我们直接处理点击事件。
        done = False
        reward = -1  # 默认的步骤惩罚
        if action[2] > 0:  # 如果动作包含了点击事件
            distance = np.linalg.norm(mouse_position - self._target)
            if distance < 10:  # 判断点击是否在目标附近
                reward = 10  # 点击正确的奖励
                done = True  # 结束环境

        self.state = mouse_position  # 更新鼠标位置

        return self.state, reward, done, {}

    def render(self, mode="human"):
        """在屏幕上渲染出当前环境的视觉呈现（如果有）。"""
        # 这个方法应该能够展示当前的屏幕、鼠标位置和目标位置。
        # 由于我们在一个简化的环境中，这个方法可能什么都不做。
        pass

    def _generate_target(self):
        """随机生成目标位置。"""
        return np.random.randint(low=0, high=[self.screen_width, self.screen_height])


if __name__ == "__main__":
    # 使用这个环境
    env = MouseEnv(screen_width=800, screen_height=600)
