import torch.nn as nn
import torch
# Define the LSTM policy network
class LSTMPolicyNetwork(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, batch_size, device):
        super(LSTMPolicyNetwork, self).__init__()
        self.num_layers = 10
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_dim)
        self.lstm = nn.LSTM(input_size=self.embedding_dim+1, hidden_size=hidden_dim,
                            num_layers=self.num_layers, batch_first=True)
        
        # Initialize the hidden state
        self.reset_hidden(device)


        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.softmax = nn.Softmax(dim=-1)

        # Sample forward pass instr: output, hidden = self.lstm(input_tensor, hidden)
    
    def reset_hidden(self, device):
        self.hidden_state = torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).to(device)
        self.cell_state = torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).to(device)




    def forward(self, input_indices, similarities):
        # Forward pass through embedding layer, LSTM, and final linear layer
        embedded = self.embedding(input_indices)

        hidden = (self.hidden_state, self.cell_state)
        embedded_indices_with_similarities = torch.cat((embedded, similarities.unsqueeze(1)), 1)

        # Fix dimensionality
        embedded_indices_with_similarities = embedded_indices_with_similarities.unsqueeze(1)

        # print(embedded.shape)
        # print(similarities.shape)
        # print(embedded_indices_with_similarities.shape)

        output, hidden = self.lstm(embedded_indices_with_similarities, hidden)

        self.hidden_state = hidden[0]
        self.cell_state = hidden[1]

        logits = self.fc(output)
        # Use only the last time step's output for decision making
        return self.softmax(logits)