import pyautogui
import numpy as np


class MouseKeyboardEnv:
    def __init__(self):
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

        return state

    def step(self, action):
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

        # Check if goal is reached
        done = current_x == 100 and current_y == 100

        return next_state, reward, done

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
        next_state, reward, done = env.step(action)
        env.render()

        if done:
            print("Goal reached!")
            break
