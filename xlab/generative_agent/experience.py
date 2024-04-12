from xlab.generative_agent.state import Perception


class Experience:
    def __init__(self, perception: Perception, action, reward) -> None:
        self.perception = perception
        self.action = action
        self.reward = reward
