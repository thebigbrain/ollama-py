from xlab.memorystream.memory_stream import MemoryStreamModule
from xlab.memorystream.perception import PerceptionModule
from xlab.memorystream.agent import Agent
from xlab.memorystream.environment import Environment


if __name__ == "__main__":
    # Create an environment and agent modules
    environment = Environment()
    agent = Agent(PerceptionModule(environment), MemoryStreamModule())

    agent.learn(environment)
