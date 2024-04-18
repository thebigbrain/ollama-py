from xlab.generative_agent.environment import Environment


class PerceptionModule:
    def __init__(self, environment: Environment):
        self.environment = environment

    def perceive(self):
        # Gather sensory data from the environment
        return self.environment.state
