import pyautogui
import numpy as np

from xlab.generative_agent.environment import Environment


class MouseKeyboardEnv(Environment):
    def __init__(self):
        super().__init__()

        self.screen_width = pyautogui.size()[0]
        self.screen_height = pyautogui.size()[1]

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

    def get_num_states(self) -> int:
        return int(self.screen_width) * int(self.screen_height)

    def reset(self):
        # Get current mouse position
        current_x, current_y = pyautogui.position()

        # Define initial state
        state = np.array(
            [
                current_x / self.screen_width,
                current_y / self.screen_height,
            ]
        )

        self.state = state

        return state

    def take_step(self, action):
        # Extract action components
        action_name, dx, dy, click = action

        # Update mouse position
        new_x = pyautogui.position()[0] + dx
        new_y = pyautogui.position()[1] + dy

        # Clip mouse position within screen boundaries
        new_x = max(0, min(new_x, self.screen_width))
        new_y = max(0, min(new_y, self.screen_height))

        # Perform mouse action
        pyautogui.moveTo(new_x, new_y)
        if click:
            pyautogui.click(button=action_name)

        # Get current mouse position
        current_x, current_y = pyautogui.position()

        # Calculate reward (assuming some goal at (100, 100))
        reward = -np.sqrt((current_x - 100) ** 2 + (current_y - 100) ** 2)

        # Define next state
        next_state = np.array(
            [
                current_x / self.screen_width,
                current_y / self.screen_height,
            ]
        )

        return next_state, reward

    def is_episode_over(self):
        # Determine if the episode has ended
        current_x = self.state[0]
        current_y = self.state[1]
        return (
            abs(current_x - 100 / self.screen_width) < 0.001
            and abs(current_y - 100 / self.screen_height) < 0.001
        )

    def render(self):
        # Simulate visual rendering (replace with your visualization method)
        print("Current mouse position:", pyautogui.position())


if __name__ == "__main__":
    env = MouseKeyboardEnv()

    # Example usage
    for _ in range(100):
        state = env.reset()
        i = np.random.choice(len(env.action_space))
        action = env.action_space[i]
        next_state, reward, done = env.take_step(action)
        env.render()

        if env.is_episode_over():
            print("Goal reached!")
            break
