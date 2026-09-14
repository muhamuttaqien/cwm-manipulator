import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class A2CAgent:
    def __init__(self, state_dim, action_dim, gamma=0.99, lr=3e-4, update_timestep=20, value_coef=0.5, entropy_coef=0.01):
        self.gamma = gamma
        self.update_timestep = update_timestep
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.timestep = 0

        self.actor = ActorNetwork(state_dim, action_dim).to(device)
        self.critic = CriticNetwork(state_dim).to(device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.memory = Memory()

    def select_action(self, state):
        state = torch.FloatTensor(state).to(device)

        action_probs = self.actor(state)
        state_value = self.critic(state)

        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()

        self.memory.states.append(state)
        self.memory.actions.append(action)
        self.memory.log_probs.append(action_dist.log_prob(action))
        self.memory.values.append(state_value.squeeze())
        self.memory.entropies.append(action_dist.entropy())

        return action.item()

    def update(self):
        # Calculate discounted returns
        returns = []
        discounted_reward = 0

        for reward, is_terminal in zip(
            reversed(self.memory.rewards),
            reversed(self.memory.is_terminals)
        ):
            if is_terminal:
                discounted_reward = 0

            discounted_reward = reward + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)

        returns = torch.tensor(returns, dtype=torch.float32).to(device)

        # Stack memory
        log_probs = torch.stack(self.memory.log_probs)
        values = torch.stack(self.memory.values)
        entropies = torch.stack(self.memory.entropies)

        # Calculate advantages
        advantages = returns - values.detach()

        # Actor loss
        actor_loss = -(log_probs * advantages).mean()
        actor_loss = actor_loss - self.entropy_coef * entropies.mean()

        # Critic loss
        critic_loss = nn.functional.mse_loss(values, returns)

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
        self.actor_optimizer.step()

        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
        self.critic_optimizer.step()

        # Clear memory
        self.memory.clear_memory()


class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorNetwork, self).__init__()

        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.action_layer = nn.Linear(64, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        action_probs = torch.softmax(self.action_layer(x), dim=-1)

        return action_probs


class CriticNetwork(nn.Module):
    def __init__(self, state_dim):
        super(CriticNetwork, self).__init__()

        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.value_layer = nn.Linear(64, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        state_value = self.value_layer(x)

        return state_value


class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.is_terminals = []
        self.entropies = []

    def clear_memory(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.is_terminals.clear()
        self.entropies.clear()
        