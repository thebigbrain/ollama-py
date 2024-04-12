from xlab.generative_agent.action import ActionPolicy
from xlab.generative_agent.environment import Environment
from xlab.generative_agent.memory_stream import Experience, MemoryStreamModule
from xlab.generative_agent.perception import PerceptionModule


class Agent:
    def __init__(
        self,
        perception_module: PerceptionModule,
        memory_stream_module: MemoryStreamModule,
        action_policy: ActionPolicy,
    ):
        self.perception_module = perception_module
        self.memory_stream_module = memory_stream_module
        self.action_policy = action_policy

    def perceive(self):
        # Perceive the current state of the environment
        state = self.perception_module.perceive()
        return state

    def take_action(self, memories, state):
        # Select an action based on memories and current state
        # This can be implemented using various decision-making algorithms, such as Q-learning or policy gradient methods
        return self.action_policy.take_action(memories, state)

    def form_long_term_plan(self):
        return self.memory_stream_module.form_long_term_plan()

    def reflect(self):
        return self.memory_stream_module.reflect()

    def learn(self, environment: Environment):
        num_episodes = 1000  # Total training episodes
        max_steps_per_episode = 200  # Maximum steps per episode

        # Run the agent in the environment and learn from experiences
        for episode in range(num_episodes):
            for step in range(max_steps_per_episode):
                # Agent perceives the environment
                state = self.perceive()

                # Agent retrieves relevant memories
                memories = self.memory_stream_module.retrieve_memories(state)

                # Agent takes an action based on memories and current state
                action = self.take_action(memories, state)

                # Environment updates and provides reward
                new_state, reward = environment.take_step(action)

                # Agent updates its memory stream

                self.memory_stream_module.add_experience(
                    Experience(state, action, reward)
                )

                # Agent forms a long-term plan and reflects
                self.form_long_term_plan()
                self.reflect()

                # Agent transitions to the new state
                state = new_state

                # Check if episode is over
                if environment.is_episode_over():
                    break
