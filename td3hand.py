from Env import CompressionEnv
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

env = CompressionEnv()


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 1)

    def forward(self, state, action):
        state = state.float()
        action = action.float()
        x = torch.cat([state, action], 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        # Scale output to the action space by multiplying by the maximum action value element by element
        x = torch.mul(x, torch.from_numpy(self.max_action))
        
        return x
    
class ReplayBuffer:
    def __init__(self, max_size=1000000):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def size(self):
        return len(self.buffer)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = env.action_space.high

policy_net = PolicyNetwork(state_dim, action_dim, max_action)
q_net1 = QNetwork(state_dim, action_dim)
q_net2 = QNetwork(state_dim, action_dim)

target_policy_net = PolicyNetwork(state_dim, action_dim, max_action)
target_q_net1 = QNetwork(state_dim, action_dim)
target_q_net2 = QNetwork(state_dim, action_dim)

# Copy weights from the original networks to the target networks
target_policy_net.load_state_dict(policy_net.state_dict())
target_q_net1.load_state_dict(q_net1.state_dict())
target_q_net2.load_state_dict(q_net2.state_dict())

policy_optimizer = optim.Adam(policy_net.parameters(), lr=3e-4)
q_optimizer1 = optim.Adam(q_net1.parameters(), lr=3e-4)
q_optimizer2 = optim.Adam(q_net2.parameters(), lr=3e-4)

def td3_train(replay_buffer, batch_size=100, gamma=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_delay=2):
    # Sample from replay buffer
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
    states = torch.from_numpy(states)
    actions = torch.from_numpy(actions)
    rewards = torch.from_numpy(rewards).unsqueeze(1)
    next_states = torch.from_numpy(next_states)
    dones = torch.from_numpy(dones).unsqueeze(1).long()

    # Compute target actions with noise
    noise = actions.data.normal_(0, policy_noise)
    noise = noise.clamp(-noise_clip, noise_clip)
    next_actions = target_policy_net(next_states) + noise
    next_actions = next_actions.clamp(torch.from_numpy(-max_action), torch.from_numpy(max_action))

    # Compute target Q values
    target_q1 = target_q_net1(next_states, next_actions).float()
    target_q2 = target_q_net2(next_states, next_actions).float()
    target_q = torch.min(target_q1, target_q2).float()
    target_q = rewards.float() + ((1 - dones) * gamma * target_q).detach().float()

    # Update Q-networks
    current_q1 = q_net1(states, actions).float()
    current_q2 = q_net2(states, actions).float()
    q_loss1 = nn.MSELoss()(current_q1, target_q).float()
    q_loss2 = nn.MSELoss()(current_q2, target_q).float()
    q_optimizer1.zero_grad()
    q_optimizer2.zero_grad()
    q_loss1.backward()
    q_loss2.backward()
    q_optimizer1.step()
    q_optimizer2.step()

    # Delayed policy updates
    if global_step % policy_delay == 0:
        # Compute policy loss
        policy_loss = -q_net1(states, policy_net(states)).mean()
        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()

        # Update target networks
        for param, target_param in zip(policy_net.parameters(), target_policy_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for param, target_param in zip(q_net1.parameters(), target_q_net1.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for param, target_param in zip(q_net2.parameters(), target_q_net2.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

replay_buffer = ReplayBuffer()

num_episodes = 1000
batch_size = 100
global_step = 0
print(action_dim, env.action_space.high)
for episode in range(num_episodes):
    state, _ = env.reset()
    original = 0
    ratio = 0

    while True:
        action = policy_net(torch.FloatTensor(state).unsqueeze(0)).detach().numpy()[0]
        next_state, reward, done, _, info = env.step(action)
        replay_buffer.add(state, action, reward, next_state, done)

        state = next_state
        original += 32 * 64
        ratio += len(info['block'])
        global_step += 1

        if replay_buffer.size() > batch_size:
            td3_train(replay_buffer, batch_size)

        if done:
            break

    print(f"Episode {episode + 1}, Compression Ratio: {original / ratio}, timestmp_ratio: {env.timestamps_ratio}")