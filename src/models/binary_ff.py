from torch import nn
import torch

# Generic Feedforward Policy Network
class BinaryFeedForwardPolicyNetwork(nn.Module):
    def __init__(self):
        super(BinaryFeedForwardPolicyNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, 2),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, input_tensor, similarity):
        combined = torch.stack((input_tensor, similarity), 1)

        return self.layers(combined)