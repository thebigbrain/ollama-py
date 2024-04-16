from xlab.generative_agent.action import Action, ActionPolicy
from xlab.generative_agent.environment import Environment
from xlab.generative_agent.memory_stream import Experience, MemoryStreamModule
from xlab.generative_agent.perception import PerceptionModule


class Agent:
    def __init__(
        self,
        environment: Environment,
        perception_module: PerceptionModule,
        memory_stream_module: MemoryStreamModule,
        action_policy: ActionPolicy,
    ):
        self.environment = environment
        self.perception_module = perception_module
        self.memory_stream_module = memory_stream_module
        self.action_policy = action_policy

    def learn(self, num_episodes=1000, max_steps_per_episode=200):
        # Run the agent in the environment and learn from experiences
        for episode in range(num_episodes):
            # Agent perceives the environment
            state = self.environment.get_current_state()
            print("episode", episode)

            for step in range(max_steps_per_episode):
                # Agent retrieves relevant memories
                memories = self.memory_stream_module.retrieve_memories(state)

                # Agent takes an action based on memories and current state
                action = self.action_policy.take_action(memories, state)

                # Environment updates and provides reward
                new_state, reward = self.environment.take_step(action)

                # Agent updates its memory stream
                self.memory_stream_module.add_experience(
                    Experience(state, action, reward, new_state)
                )

                # Agent forms a long-term plan and reflects
                self.memory_stream_module.form_long_term_plan()
                self.memory_stream_module.reflect()

                # Agent transitions to the new state
                state = new_state

                # Check if an episode is over
                if self.environment.is_episode_over():
                    break


def create_agent(
    environment: Environment,
    perception_module: PerceptionModule,
    memory_stream_module: MemoryStreamModule,
    action_policy: ActionPolicy,
):
    return Agent(
        environment=environment,
        perception_module=perception_module,
        memory_stream_module=memory_stream_module,
        action_policy=action_policy,
    )
