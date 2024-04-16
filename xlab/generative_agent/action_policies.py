import numpy as np
from xlab.generative_agent.action import Action, ActionPolicy
from xlab.generative_agent.environment import Environment
from xlab.generative_agent.state import EnvState


class EGreedyPolicy(ActionPolicy):
    epsilon = 0.1

    def __init__(self, environment: Environment, alpha=0.2, gamma=0.9, epsilon=0.1):
        super().__init__()
        self.environment = environment
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # Initialize Q-table with zeros
        self.Q_table = np.zeros(
            (environment.get_available_states(), environment.get_available_actions())
        )

    def take_action(self, memories, state: EnvState) -> Action:
        """
        Selects an action based on memories and current state using Q-learning.

        Args:
            memories: A list of retrieved memories from the memory stream.
            state: The current state of the environment represented as a NumPy array.

        Returns:
            The selected action.
        """

        # Extract relevant information from memories (optional)
        # You can potentially analyze past rewards and actions from memories to inform decision-making.

        # Epsilon-greedy exploration
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.environment.get_available_actions())

        # Select action with highest Q-value
        q_values = self.Q_table[state]
        best_action = np.argmax(q_values)
        return best_action

    def update(self, state, action, reward, next_state):
        # Q-learning update rule
        current_q_value = self.Q_table[state, action]
        next_q_value_max = np.max(self.Q_table[next_state])
        new_q_value = current_q_value + self.alpha * (
            reward + self.gamma * next_q_value_max - current_q_value
        )
        self.Q_table[state, action] = new_q_value


class PolicyGradientActionPolicy(ActionPolicy):
    def take_action(self, memories, state) -> Action:
        """
        Selects an action based on memories and current state using a policy gradient method.

        Args:
            memories: A list of retrieved memories from the memory stream.
            state: The current state of the environment represented as a NumPy array.

        Returns:
            The selected action.
        """

        # Extract relevant information from memories (optional)
        # You can potentially analyze past rewards and actions from memories to inform decision-making.

        # Access the policy network (assuming it's a neural network stored within the agent class)
        policy_network = self.policy_network

        # Get the probability distribution for actions (output from the policy network)
        action_probs = policy_network.predict(np.array([state]))[0]

        # Sample an action based on the probability distribution
        action = np.random.choice(
            self.environment.get_available_actions(), p=action_probs
        )

        return action
