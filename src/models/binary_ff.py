from torch import nn
import torch

# Generic Feedforward Policy Network
class BinaryFeedForwardPolicyNetwork(nn.Module):
    def __init__(self):
        super(BinaryFeedForwardPolicyNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(3, 4),
            nn.ReLU(),
            nn.Linear(4, 2),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, input_tensor, similarity):

        # Make sure input tensor has width of 2..
        assert input_tensor.shape[-1] == 2, "Input tensor must have width of 2"

        combined = torch.cat((input_tensor, similarity.unsqueeze(1)), 1)
        return self.layers(combined)