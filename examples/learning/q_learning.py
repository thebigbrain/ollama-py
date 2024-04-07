import numpy as np

from examples.learning.reinforcement.key_mouse_env import KeyMouseBehaviourEnv

env = KeyMouseBehaviourEnv()

num_episodes = 1000
alpha = 0.2
gamma = 0.96

num_states = env.observation_space.n
num_actions = env.action_space.n

# Initialize Q-table with zeros
Q = np.zeros([num_states, num_actions])


for i_episode in range(num_episodes):
    # Reset the environment
    state = env.reset()
    done = False

    while not done:
        # Choose action based on Q-table
        action = np.argmax(Q[state])
        # Execute the action and get feedback
        next_state, reward, done, info = env.step(action)
        # Update Q-table
        Q[state][action] = (1 - alpha) * Q[state][action] \
                            + alpha * (reward + gamma * np.max(Q[next_state]))
        # Update state
        state = next_state