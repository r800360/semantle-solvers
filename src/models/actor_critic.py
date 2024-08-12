import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions

class ActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_sizes=(64, 64)):
        super(ActorCritic, self).__init__()
        
        # Actor network
        self.pi_net = nn.Sequential(
            nn.Linear(input_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], output_dim),
            nn.Softmax(dim=-1)  # Output is a probability distribution over actions
        )
        
        # Critic network
        self.v_net = nn.Sequential(
            nn.Linear(input_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], 1)  # Output is a single value estimate
        )
    
    def step(self, obs):
        # Compute the action distribution from the actor network
        pi = self.pi_net(obs)
        dist = distributions.Categorical(pi)
        
        # Sample an action from the distribution
        a = dist.sample()
        
        # Compute the log probability of the chosen action
        logp_a = dist.log_prob(a)
        
        # Compute the value estimate from the critic network
        v = self.v_net(obs).squeeze()
        
        #return a.numpy(), v.numpy(), logp_a.numpy()
        return a, v, logp_a
    
    def act(self, obs):
        # Only return the action (used during evaluation)
        pi = self.pi_net(obs)
        dist = distributions.Categorical(pi)
        a = dist.sample()
        return a.numpy()
    
    def forward(self, obs):
        # The forward method can return both the policy distribution and the value
        pi = self.pi_net(obs)
        v = self.v_net(obs).squeeze()
        return pi, v
