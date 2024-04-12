class EnvState:
    pass


class Environment:
    def __init__(self):
        # Initialize the environment's state variables
        self.state: EnvState = ...

    def get_current_state(self):
        return self.state

    def take_step(self, action):
        # Update the environment based on the action
        new_state = ...
        reward = ...

        # Update the internal state
        self.state = new_state

        return new_state, reward

    def is_episode_over(self):
        # Determine if the episode has ended
        return ...