from xlab.generative_agent.action_policies import EpsilonGreedyPolicy
from xlab.generative_agent.key_mouse_env import MouseKeyboardEnv
from xlab.generative_agent.memory_stream import MemoryStreamModule
from xlab.generative_agent.perception import PerceptionModule
from xlab.generative_agent.agent import create_agent


if __name__ == "__main__":
    # Create an environment and agent modules
    environment = MouseKeyboardEnv()
    agent = create_agent(
        environment=environment,
        perception_module=PerceptionModule(environment),
        memory_stream_module=MemoryStreamModule(),
        action_policy=EpsilonGreedyPolicy(
            environment.get_num_states(), environment.get_num_actions()
        ),
    )

    agent.learn()
