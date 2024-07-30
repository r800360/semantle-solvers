import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Create the environment
env = gym.make("CartPole-v1")

# Get the state and action sizes
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Define the policy function (neural network)
class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_size, 32)
        self.fc2 = nn.Linear(32, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.softmax(x, dim=-1)

policy = Policy()

# Define the loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(policy.parameters(), lr=0.01)

# Define the update policy function
def update_policy(rewards, log_probs, optimizer):
    log_probs = torch.stack(log_probs)
    loss = -torch.mean(log_probs) * (sum(rewards) - 15)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

gamma = 0.99

# Training loop
for episode in range(5000):
    state, _ = env.reset()
    done = False
    score = 0
    log_probs = []
    rewards = []
    
    while not done:
        # Select action
        state = torch.tensor(state, dtype=torch.float32).reshape(1, -1)
        probs = policy(state)
        action = torch.multinomial(probs, 1).item()
        log_prob = torch.log(probs[:, action])

        # Take step
        next_state, reward, done, _, _ = env.step(action)
        score += reward
        
        rewards.append(reward)
        log_probs.append(log_prob)
        state = next_state

    # Update policy
    print(f"Episode {episode}: {score}")
    update_policy(rewards, log_probs, optimizer)