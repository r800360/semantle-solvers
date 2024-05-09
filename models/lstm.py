import torch.nn as nn

# Define the LSTM policy network
class LSTMPolicyNetwork(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(LSTMPolicyNetwork, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=10, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, input_indices):
        # Forward pass through embedding layer, LSTM, and final linear layer
        embedded = self.embedding(input_indices)
        output, _ = self.lstm(embedded)
        logits = self.fc(output)
        # Use only the last time step's output for decision making
        return self.softmax(logits)