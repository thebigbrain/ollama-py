from xlab.generative_agent.environment import Environment


class PerceptionModule:
    def __init__(self, environment: Environment):
        self.environment = environment

    def perceive(self):
        # Gather sensory data from the environment
        state = self.environment.get_current_state()
        return state
