import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import List, Tuple


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class ValueNetwork(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super(ValueNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class PPOAgent:
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        clip_epsilon: float = 0.2
    ) -> None:
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.policy_net = PolicyNetwork(input_dim, output_dim)
        self.value_net = ValueNetwork(input_dim)
        self.optimizer = optim.Adam(list(self.policy_net.parameters()) + list(self.value_net.parameters()), lr=lr)

    def select_action(self, state: torch.Tensor) -> Tuple[int, torch.Tensor]:
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy_net(state)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def compute_advantages(self, rewards: List[float], values: List[float], dones: List[bool]) -> Tuple[
        torch.Tensor, torch.Tensor]:
        advantages = []
        returns = []
        gae = 0.0
        value_next = 0.0

        for i in reversed(range(len(rewards))):
            if dones[i]:
                value_next = 0
                gae = 0

            delta = rewards[i] + self.gamma * value_next - values[i]
            gae = delta + self.gamma * gae
            advantages.insert(0, gae)
            value_next = values[i]
            returns.insert(0, gae + values[i])

        return torch.tensor(advantages, dtype=torch.float32), torch.tensor(returns, dtype=torch.float32)

    def update(
        self,
        states: List[List[float]],
        actions: List[int],
        log_probs: List[float],
        returns: torch.Tensor,
        advantages: torch.Tensor
    ) -> None:
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        old_log_probs = torch.FloatTensor(log_probs)
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)

        for _ in range(4):  # Run multiple epochs for stability
            new_probs = self.policy_net(states)
            dist = Categorical(new_probs)
            new_log_probs = dist.log_prob(actions)

            ratio = torch.exp(new_log_probs - old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
            policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

            value_loss = nn.MSELoss()(self.value_net(states).squeeze(), returns)

            self.optimizer.zero_grad()
            (policy_loss + 0.5 * value_loss).backward()
            self.optimizer.step()
