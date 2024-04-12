import numpy as np


class Action:
    pass


class ActionPolicy:
    def take_action(self, memories, state) -> Action:
        raise NotImplemented("action policy not implemented")


class QLearningPolicy(ActionPolicy):
    def take_action(self, memories, state) -> Action:
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

        # Access the Q-value table (assuming it's stored within the agent class)
        Q_table = self.Q_table

        # Get all possible actions (replace with your environment's action space definition)
        possible_actions = self.environment.get_available_actions()

        # Select the action with the highest Q-value for the current state
        best_action = np.argmax(Q_table[state])

        # Epsilon-greedy exploration (optional)
        if np.random.rand() < self.epsilon:  # Explore with a small probability
            best_action = np.random.choice(possible_actions)

        return best_action


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
