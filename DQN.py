"""
Deep Q-Network in PyTorch
------------------------------------------------
* Experience replay with burn-in
* Fixed target network
* Epsilon-greedy exploration (exponential decay)
* Double-DQN (optional)
* Periodic evaluation (greedy policy)
"""
import argparse
import random
import sys
from collections import deque
from pathlib import Path
from typing import List, Tuple, NamedTuple, Optional

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gym.wrappers import RecordVideo
from torch.nn import functional as F


# Network
class QNetwork(nn.Module):
    """Simple MLP Q-network."""
    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# Transition tuple
class Transition(NamedTuple):
    state: np.ndarray
    action: int
    reward: float
    done: bool
    next_state: np.ndarray

# Replay buffer
class ReplayBuffer:
    def __init__(self, capacity: int = 50_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)

# DQN Agent
class DQNAgent:
    def __init__(
        self,
        env: gym.Env,
        *,
        hidden_dim: int = 128,
        buffer_capacity: int = 50_000,
        burn_in: int = 10_000,
        batch_size: int = 32,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.999,
        lr: float = 1e-3,
        target_update_freq: int = 1_000,
        double_dqn: bool = True,
        device: Optional[torch.device] = None,
    ):
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.n_actions = env.action_space.n
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_net = QNetwork(self.obs_dim, self.n_actions, hidden_dim).to(self.device)
        self.target_net = QNetwork(self.obs_dim, self.n_actions, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.double_dqn = double_dqn
        self.step_counter = 0

        self.memory = ReplayBuffer(buffer_capacity)
        self.burn_in = burn_in

        self.eval_every = 50
        self.eval_episodes = 20
        self.best_avg_reward = -np.inf
        self.reward_history: List[float] = []
        self.eval_episode_nums: List[int] = []

    
    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        """Epsilon-greedy action selection."""
        if random.random() < epsilon:
            return self.env.action_space.sample()

        state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_t)
        return int(q_values.argmax(dim=1).item())

    def _decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    
    def burn_in_memory(self) -> None:
        print(f"Burn-in: filling replay buffer with {self.burn_in} transitions...")
        state = self.env.reset()
        for _ in range(self.burn_in):
            action = self.env.action_space.sample()
            next_state, reward, done, _ = self.env.step(action)
            self.memory.push(
                Transition(state, action, reward, done, next_state)
            )
            state = next_state if not done else self.env.reset()
        print("Burn-in completed.")

    
    def update(self) -> float:
        if len(self.memory) < self.batch_size:
            return 0.0

        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, dones, next_states = zip(*batch)

        states_t = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions_t = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states_t = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)

        current_q = self.q_net(states_t).gather(1, actions_t)

        with torch.no_grad():
            if self.double_dqn:
                next_actions = self.q_net(next_states_t).argmax(dim=1, keepdim=True)
                next_q = self.target_net(next_states_t).gather(1, next_actions)
            else:
                next_q = self.target_net(next_states_t).max(dim=1, keepdim=True)[0]

            target_q = rewards_t + self.gamma * next_q * (1 - dones_t)

        loss = F.mse_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_counter += 1
        if self.step_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

    def evaluate(self) -> float:
        rewards = []
        for _ in range(self.eval_episodes):
            state = self.env.reset()
            episode_reward = 0.0
            done = False
            while not done:
                action = self.select_action(state, epsilon=0.0) 
                state, reward, done, _ = self.env.step(action)
                episode_reward += reward
            rewards.append(episode_reward)
        avg_reward = float(np.mean(rewards))
        std_reward = float(np.std(rewards))
        print(f"Evaluation -> Avg reward: {avg_reward:.2f} ± {std_reward:.2f}")
        return avg_reward

    def train(self, total_episodes: int = 5000, render: bool = False, save_dir: Optional[Path] = None) -> None:
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            self.env = RecordVideo(self.env, save_dir, episode_trigger=lambda ep: ep % 500 == 0)

        self.burn_in_memory()

        recent_rewards = deque(maxlen=100)

        for episode in range(1, total_episodes + 1):
            state = self.env.reset()
            episode_reward = 0.0
            done = False

            while not done:
                if render:
                    self.env.render()

                action = self.select_action(state, self.epsilon)
                next_state, reward, done, _ = self.env.step(action)

                self.memory.push(Transition(state, action, reward, done, next_state))

                loss = self.update()
                state = next_state
                episode_reward += reward

                self._decay_epsilon()

            recent_rewards.append(episode_reward)

            if episode % 10 == 0:
                avg_recent = np.mean(recent_rewards)
                print(f"Episode {episode:4d} | Reward {episode_reward:6.1f} | "
                      f"Running Avg {avg_recent:6.1f} | ε {self.epsilon:.3f}")

        
            if episode % self.eval_every == 0:
                avg_eval = self.evaluate()
                self.eval_episode_nums.append(episode)
                self.reward_history.append(avg_eval)

                if avg_eval > self.best_avg_reward and save_dir:
                    path = save_dir / "best_model.pth"
                    torch.save(self.q_net.state_dict(), path)
                    print(f"New best model saved (avg reward {avg_eval:.2f}) -> {path}")
                    self.best_avg_reward = avg_eval

        self.env.close()
        self._plot_evaluation()

    def _plot_evaluation(self) -> None:
        if not self.reward_history:
            return
        plt.figure(figsize=(8, 5))
        plt.title("DQN Evaluation")
        plt.xlabel("Training Episode")
        plt.ylabel(f"Average Reward over {self.eval_episodes} episodes")
        plt.plot(self.eval_episode_nums, self.reward_history, marker="o")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def load_model(self, path: Path) -> None:
        self.q_net.load_state_dict(torch.load(path, map_location=self.device))
        self.q_net.eval()
        self.target_net.load_state_dict(self.q_net.state_dict())
        print(f"Model loaded from {path}")

    def test(self, episodes: int = 20, render: bool = False) -> None:
        rewards = []
        for ep in range(episodes):
            state = self.env.reset()
            total = 0.0
            done = False
            while not done:
                if render:
                    self.env.render()
                action = self.select_action(state, epsilon=0.0)
                state, reward, done, _ = self.env.step(action)
                total += reward
            rewards.append(total)
            print(f"Test Episode {ep+1:2d} | Reward: {total:.1f}")
        print(f"Test Avg: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DQN for Gym environments")
    parser.add_argument("--env", type=str, required=True, help="Gym environment ID")
    parser.add_argument("--episodes", type=int, default=5000, help="Total training episodes")
    parser.add_argument("--render", action="store_true", help="Render training episodes")
    parser.add_argument("--save-dir", type=str, default=None, help="Directory to save videos & best model")
    parser.add_argument("--model", type=str, default=None, help="Path to a .pth model for testing only")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--double", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    env = gym.make(args.env)
    env.seed(seed)

    save_dir = Path(args.save_dir) if args.save_dir else None

    agent = DQNAgent(
        env,
        double_dqn=args.double,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    if args.model:
        model_path = Path(args.model)
        if not model_path.is_file():
            print(f"Model file not found: {model_path}")
            sys.exit(1)
        agent.load_model(model_path)
        agent.test(episodes=20, render=args.render)
    else:
        agent.train(
            total_episodes=args.episodes,
            render=args.render,
            save_dir=save_dir,
        )


if __name__ == "__main__":
    main()