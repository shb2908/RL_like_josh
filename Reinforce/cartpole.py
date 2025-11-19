import torch
import torch.nn as nn
import torch.optim as optim
import gym
import numpy as np
from collections import deque

LEARNING_RATE = 1e-2
GAMMA = 0.99
HIDDEN_SIZE = 128
EPISODES = 500

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.fc(x)

def compute_returns(rewards, gamma):
    R = 0
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns

def train():
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    policy = PolicyNetwork(state_dim, action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)
    
    scores = deque(maxlen=100)

    for episode in range(1, EPISODES + 1):
        state, _ = env.reset()
        log_probs = []
        rewards = []
        
        # (Sample {tau} from pi_theta)
        done = False
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            probs = policy(state_tensor)
            
            m = torch.distributions.Categorical(probs)
            action = m.sample()
            
            log_probs.append(m.log_prob(action))
            
            state, reward, done, truncated, _ = env.step(action.item())
            done = done or truncated
            rewards.append(reward)
            
        scores.append(sum(rewards))
        
        # Compute Returns
        returns = compute_returns(rewards, GAMMA)
        
        # Policy Gradient Loss
        loss = 0
        for log_prob, R in zip(log_probs, returns):
            loss -= log_prob * R 
            
        # Update Policy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if episode % 50 == 0:
            print(f"Episode {episode}\tAverage Score: {np.mean(scores):.2f}")
            
        if np.mean(scores) >= 495.0:
            print(f"Solved in {episode} episodes!")
            break

    env.close()

if __name__ == "__main__":
    train()