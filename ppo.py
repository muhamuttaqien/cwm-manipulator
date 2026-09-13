import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PPOAgent:
    def __init__(self, state_dim, action_dim, clip_epsilon=0.2, gamma=0.99, lr=3e-4, update_timestep=2000):
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma
        self.update_timestep = update_timestep
        self.timestep = 0

        self.policy = PolicyNetwork(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.policy_old = PolicyNetwork(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.mse_loss = nn.MSELoss()
        self.memory = Memory()

    def select_action(self, state):
        state = torch.FloatTensor(state).to(device)
        action_probs = self.policy_old(state)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        self.memory.states.append(state)
        self.memory.actions.append(action)
        self.memory.log_probs.append(action_dist.log_prob(action))
        return action.item()

    def update(self):
        # Calculate discounted rewards
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.memory.rewards), reversed(self.memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + self.gamma * discounted_reward
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        old_states = torch.stack(self.memory.states).detach().to(device)
        old_actions = torch.stack(self.memory.actions).detach().to(device)
        old_log_probs = torch.stack(self.memory.log_probs).detach().to(device)

        for _ in range(4):  # Multiple epochs for PPO update
            log_probs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # Find the ratio (pi_theta / pi_theta_old)
            ratios = torch.exp(log_probs - old_log_probs.detach())

            # Calculate surrogate loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.mse_loss(state_values, rewards) - 0.01 * dist_entropy

            # Update the policy
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Clear memory
        self.memory.clear_memory()

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.action_layer = nn.Linear(64, action_dim)
        self.value_layer = nn.Linear(64, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        action_probs = torch.softmax(self.action_layer(x), dim=-1)
        return action_probs

    def evaluate(self, state, action):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        action_probs = torch.softmax(self.action_layer(x), dim=-1)
        state_values = self.value_layer(x)
        action_dist = torch.distributions.Categorical(action_probs)
        action_log_probs = action_dist.log_prob(action)
        dist_entropy = action_dist.entropy()
        return action_log_probs, state_values, dist_entropy

class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.states[:]
        del self.actions[:]
        del self.log_probs[:]
        del self.rewards[:]
        del self.is_terminals[:]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
