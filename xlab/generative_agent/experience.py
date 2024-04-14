from xlab.generative_agent.state import Perception, EnvState
from xlab.generative_agent.action import Action


class Experience:
    def __init__(self, perception: Perception, action: Action, reward: float, next_state: EnvState) -> None:
        self.perception = perception
        self.action = action
        self.reward = reward
        self.next_state = next_state
