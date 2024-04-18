import pyautogui
import numpy as np

from xlab.generative_agent.action import Action
from xlab.generative_agent.environment import Environment
from xlab.generative_agent.grid import calculate_grid_size


class MouseKeyboardEnv(Environment):
    position = None

    def __init__(self):
        super().__init__()

        self.screen_width, self.screen_height = pyautogui.size()

        self.grid = (1000, 800)
        self.grid_size = calculate_grid_size(
            screen_width=self.screen_width,
            screen_height=self.screen_height,
            grid_rows=self.grid[0],
            grid_cols=self.grid[1],
        )

        self.action_space = [
            # Mouse movements
            ("move_left", -20, 0, False),
            ("move_right", 20, 0, False),
            ("move_up", 0, -20, False),
            ("move_down", 0, 20, False),
            # Mouse clicks
            ("left", 0, 0, True),
            ("right", 0, 0, False),
            # Keyboard actions
            ("press_a", ord("a"), 0, False),
            ("press_b", ord("b"), 0, False),
            ("press_space", ord(" "), 0, False),
        ]

        self.reset()

    def get_num_actions(self) -> int:
        return len(self.action_space)

    def get_num_states(self):
        return self.grid[0] * self.grid[1]

    def _get_state(self, x, y):
        grid_width, grid_height = self.grid_size

        return np.array(
            [
                int(x // grid_height),
                int(y // grid_width),
            ]
        )

    def reset(self):
        # Get current mouse position
        self.position = pyautogui.position()

        # Define initial state
        self.state = self._get_state(self.position[0], self.position[1])

    def take_step(self, action: Action):
        # Extract action components
        action_info = self.action_space[action]
        action_name, dx, dy, click = action_info

        x, y = self.position

        # Update mouse position
        new_x = x + dx
        new_y = y + dy

        # Clip mouse position within screen boundaries
        new_x = max(0, min(new_x, self.screen_width))
        new_y = max(0, min(new_y, self.screen_height))

        self.position = (new_x, new_y)

        self.state = self._get_state(new_x, new_y)

        return self.state, self.get_reward()

    def get_reward(self):
        reward = -np.sqrt((self.state[0] - 100) ** 2 + (self.state[1] - 100) ** 2)
        return reward

    def is_episode_over(self):
        # Determine if the episode has ended
        return self.state[0] == 100 and self.state[1] == 100

    def render(self):
        # Simulate visual rendering (replace with your visualization method)
        print("Current mouse position:", self.position)


if __name__ == "__main__":
    env = MouseKeyboardEnv()

    # Example usage
    for _ in range(100000):
        env.reset()
        action = np.random.choice(env.get_num_actions())
        next_state, reward = env.take_step(action)
        env.render()

        if env.is_episode_over():
            print("Goal reached!")
            break
