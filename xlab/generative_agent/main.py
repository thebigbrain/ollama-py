from xlab.generative_agent.memory_stream import MemoryStreamModule
from xlab.generative_agent.perception import PerceptionModule
from xlab.generative_agent.agent import create_agent
from xlab.generative_agent.environment import Environment


if __name__ == "__main__":
    # Create an environment and agent modules
    environment = Environment()
    agent = create_agent(
        environment=environment,
        perception_module=PerceptionModule(environment),
        memory_stream_module=MemoryStreamModule(),
    )

    agent.learn()
