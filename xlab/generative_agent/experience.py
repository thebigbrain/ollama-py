from xlab.generative_agent.state import Perception
from xlab.generative_agent.action import Action


class Experience:
    def __init__(self, perception: Perception, action: Action, reward: float) -> None:
        self.perception = perception
        self.action = action
        self.reward = reward
