from torch import nn

# Generic Feedforward Policy Network
class FeedForwardPolicyNetwork(nn.Module):
    def __init__(self, vocab_size, hidden_dim, embedding_dim):
        super(FeedForwardPolicyNetwork, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Embedding(vocab_size, embedding_dim),
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, input_tensor):
        return self.layers(input_tensor)