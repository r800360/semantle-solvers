import logging
import torch.nn as nn
import torch


logger = logging.getLogger(__name__)
# Define the LSTM policy network
class LSTMPolicyNetwork(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, batch_size, device):
        super(LSTMPolicyNetwork, self).__init__()
        self.num_layers = 2
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        
        self.input_dim = 2 # 1 for index, 1 for similarity
        # self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_dim)
        # old input size was embedding_dim+1
        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=hidden_dim,
                            num_layers=self.num_layers, batch_first=True)


        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.softmax = nn.Softmax(dim=-1)
        
        # Initialize the hidden state
        self.reset_hidden(device)

        # Sample forward pass instr: output, hidden = self.lstm(input_tensor, hidden)
    
    def reset_hidden(self, device):
        self.hidden_state = torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).to(device)
        self.cell_state = torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).to(device)
        
    def forward(self, input_indices, similarities):
        hidden = (self.hidden_state, self.cell_state)

        # Combine word indices and similarities as a 2D feature vector (input_dim = 2)
        input_data = torch.cat((input_indices.unsqueeze(-1), similarities.unsqueeze(-1)), dim=-1)

        # Reshape to match LSTM expectations (batch_size, seq_len, input_size)
        input_data = input_data.view(self.batch_size, -1, self.input_dim)

        output, hidden = self.lstm(input_data, hidden)

        # Update hidden states
        self.hidden_state = hidden[0]
        self.cell_state = hidden[1]

        # Pass through fully connected layer and apply softmax
        logits = self.fc(output[:, -1, :])  # Use the output of the last time step
        return self.softmax(logits)

    def get_action(self, state):
        # Split the state into indices and similarities
        indices = torch.tensor([state[0], state[2], state[4]])  # Assuming indices are at positions 0, 2, 4, etc.
        similarities = torch.tensor([state[1], state[3], state[5]])  # Assuming similarities are at positions 1, 3, 5, etc.

        # Pass through the network to get action probabilities
        probs = self.forward(indices, similarities)[0]

        logger.debug(f"Probs: {probs}")
        action = torch.multinomial(probs, 1).item()
        log_prob = torch.log(probs.squeeze(0)[action])
        return action, log_prob

    # def forward(self, input_indices, similarities):
    #     # Forward pass through embedding layer, LSTM, and final linear layer
    #     # embedded = self.embedding(input_indices)

    #     # hidden = (self.hidden_state, self.cell_state)
    #     # embedded_indices_with_similarities = torch.cat((embedded, similarities.unsqueeze(1)), 1)

    #     # # Fix dimensionality
    #     # embedded_indices_with_similarities = embedded_indices_with_similarities.unsqueeze(1)

    #     # # print(embedded.shape)
    #     # # print(similarities.shape)
    #     # # print(embedded_indices_with_similarities.shape)

    #     # output, hidden = self.lstm(embedded_indices_with_similarities, hidden)
        
    #     hidden = (self.hidden_state, self.cell_state)
    #     indices_with_similarities = torch.cat((input_indices, similarities.unsqueeze(1)), 1)
        
    #     output, hidden = self.lstm(indices_with_similarities, hidden)

    #     self.hidden_state = hidden[0]
    #     self.cell_state = hidden[1]

    #     logits = self.fc(output)
    #     # Use only the last time step's output for decision making
    #     return self.softmax(logits)
    
    # def get_action(self, state):
    #     # print(state)
    #     # raise NotImplementedError
    #     state = torch.FloatTensor(state).unsqueeze(0)
    #     # #state = torch.reshape(state, (1, 10))
    #     # # Calculate the number of features in the state
    #     # num_features = state.numel()  # Flatten the state and get the total number of elements
        
    #     # # Reshape the state to a 1x(num_features) tensor
    #     # state = state.view(1, num_features)
        
    #     # # Calculate padding needed to make it a 1x6 tensor
    #     # padding = 6 - num_features
        
    #     # if padding > 0:
    #     #     # Pad the state tensor with zeros on the right
    #     #     state = F.pad(state, (0, padding), "constant", 0)
            
    #     print(state)
    #     probs = self.forward(state)[0]
    #     print(probs)
    #     logger.debug(f"Probs: {probs}")
    #     action = torch.multinomial(probs, 1).item()
    #     log_prob = torch.log(probs.squeeze(0)[action])
    #     return action, log_prob