from xlab.generative_agent.action import Action
from xlab.generative_agent.state import EnvState


class Environment:
    state: EnvState

    def reset(self):
        pass

    def take_step(self, action: Action):
        # Update the environment based on the action
        new_state = ...
        reward = ...

        # Update the internal state
        self.state = new_state

        return new_state, reward

    def get_reward(self):
        pass

    def is_episode_over(self):
        # Determine if the episode has ended
        return ...

    def get_num_actions(self) -> int:
        pass

    def get_num_states(self) -> tuple[int, int]:
        pass

    def render(self):
        pass
