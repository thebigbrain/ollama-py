from xlab.memorystream.environment import EnvState


class Experience:
    def __init__(self, perception, action, reward) -> None:
        self.perception = perception
        self.action = action
        self.reward = reward
