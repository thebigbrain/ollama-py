import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

from examples.mouse_behaviour.mouse_env import MouseEnv

# 设定随机种子以便结果可重现
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


# Q-Network模型定义
class QNetwork(nn.Module):
    def __init__(self, observation_space, action_space):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(
            np.prod(observation_space.shape), 64
        )  # 使用np.prod来将observation_space.shape转换成单个数字
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_space.shape[0])

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平输入
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# 环境和网络初始化
env = MouseEnv(screen_width=800, screen_height=600)
q_network = QNetwork(env.observation_space, env.action_space)
optimizer = optim.Adam(q_network.parameters())

# Replay Memory
replay_memory = deque(maxlen=10000)

# 训练参数
num_episodes = 1000
batch_size = 64
gamma = 0.99  # Discount factor


# 选择动作 - 这里以贪心策略为例，随机选择动作或者根据网络预测选择
def select_action(state, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()
    else:
        with torch.no_grad():
            return q_network(torch.from_numpy(state).float()).max(0)[1].view(1, 1)


# 训练步骤
def optimize_model():
    if len(replay_memory) < batch_size:
        return

    transitions = random.sample(replay_memory, batch_size)
    batch = np.array(transitions, dtype=object).transpose()

    state_batch = torch.cat(batch[0])
    action_batch = torch.cat(batch[1])
    reward_batch = torch.cat(batch[2])
    next_state_batch = torch.cat([s for s in batch[3] if s is not None])
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch[3])), dtype=torch.bool
    )

    state_action_values = q_network(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(batch_size)
    next_state_values[non_final_mask] = q_network(next_state_batch).max(1)[0].detach()
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    loss = nn.functional.smooth_l1_loss(
        state_action_values, expected_state_action_values.unsqueeze(1)
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# 训练循环
for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0

    while True:
        epsilon = 0.05  # 可适当调节epsilon以进行探索
        action = select_action(np.array([env.state]), epsilon)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward

        replay_memory.append(
            (
                torch.from_numpy(np.array([state])).float(),
                torch.tensor([[action]], dtype=torch.long),
                torch.tensor([reward], dtype=torch.float),
                torch.from_numpy(np.array([next_state])).float() if not done else None,
            )
        )

        if done:
            print(f"Episode {episode} finished with reward {total_reward}")
            break

        state = next_state
        optimize_model()

print("Training complete")

# 保存训练的模型
torch.save(q_network.state_dict(), "mouse_behaviour.pth")
