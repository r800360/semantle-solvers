from torch import nn
import torch

# Generic Feedforward Policy Network
class FeedForwardPolicyNetwork(nn.Module):
    def __init__(self, vocab_size, hidden_dim, embedding_dim):
        super(FeedForwardPolicyNetwork, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim+1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, input_tensor, similarity):
        embedded = self.embedding(input_tensor)
        combined = torch.cat((embedded, similarity.unsqueeze(1)), 1)

        return self.layers(combined)