from xlab.generative_agent.memory_stream import MemoryStreamModule
from xlab.generative_agent.perception import PerceptionModule
from xlab.generative_agent.agent import Agent
from xlab.generative_agent.environment import Environment


if __name__ == "__main__":
    # Create an environment and agent modules
    environment = Environment()
    agent = Agent(PerceptionModule(environment), MemoryStreamModule())

    agent.learn(environment)
