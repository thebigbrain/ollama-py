from xlab.generative_agent.action import Action
from xlab.generative_agent.state import EnvState


class Environment:
    state: EnvState

    def reset():
        pass

    def get_current_state(self):
        return self.state

    def take_step(self, action: Action):
        # Update the environment based on the action
        new_state = ...
        reward = ...

        # Update the internal state
        self.state = new_state

        return new_state, reward

    def is_episode_over(self):
        # Determine if the episode has ended
        return ...

    def get_available_actions(self) -> int:
        pass

    def get_available_states(self) -> int:
        pass
