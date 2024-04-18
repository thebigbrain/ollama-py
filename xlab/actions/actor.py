import torch
import torch.nn as nn
import torch.optim as optim


class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
            nn.Tanh(),  # 假设动作输出被标准化到[-1, 1]
        )

    def forward(self, state):
        return self.network(state)


class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=256):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size + action_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),  # 输出单一值，表示动作的价值
        )

    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)  # 将状态和动作拼接
        return self.network(state_action)


if __name__ == "__main__":
    state_size = 10  # 示例状态维度
    action_size = 2  # 示例动作维度

    actor = Actor(state_size, action_size)
    critic = Critic(state_size, action_size)

    actor_optimizer = optim.Adam(actor.parameters(), lr=1e-4)
    critic_optimizer = optim.Adam(critic.parameters(), lr=1e-3)

    for episode in range(num_episodes):
        state = env.reset()  # 重置环境，获取初始状态信息
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)  # 转换为Tensor
            action = actor(state_tensor)  # 生成动作

            next_state, reward, done, _ = env.step(
                action.detach().numpy()
            )  # 执行动作，获取下一个状态和奖励

            # 计算预期的Q值
            with torch.no_grad():
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
                expected_Q = reward + gamma * critic(
                    next_state_tensor, actor(next_state_tensor)
                )

            # Critic更新
            Q = critic(state_tensor, action)
            critic_loss = F.mse_loss(Q, expected_Q)
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # Actor更新
            actor_loss = -critic(state_tensor, actor(state_tensor)).mean()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            state = next_state
