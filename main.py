import sqlite3
from numpy.linalg import norm
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim

word2vec = {}
for letter_range in ("a-c", "d-h", "i-o", "p-r", "s-z"):
    with sqlite3.connect(f"./data/word2vec_{letter_range}.db") as con:
        cur = con.execute("SELECT * FROM word2vec")
        for word, vec in cur:
            vec = np.frombuffer(vec, dtype=np.float32)
            word2vec[word] = vec / norm(vec)

def similarity_function(first_word, second_word):
    x = word2vec[first_word]
    y = word2vec[second_word]
    return np.dot(x, y)/(np.linalg.norm(x) * np.linalg.norm(y) )

# Define the LSTM policy network
class LSTMPolicyNetwork(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(LSTMPolicyNetwork, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, input_indices):
        # Forward pass through embedding layer, LSTM, and final linear layer
        embedded = self.embedding(input_indices)
        output, _ = self.lstm(embedded)
        logits = self.fc(output)
        # Use only the last time step's output for decision making
        return self.softmax(logits[:, -1, :])

# Reinforcement learning training loop
def train_lstm_rl_policy(target_word, vocab, model, episodes, max_steps):
    optimizer = optim.AdamW(model.parameters())
    loss_fn = nn.NLLLoss()  # Negative log likelihood loss

    for episode in range(episodes):
        # Initialize the state (history of words and similarity scores)
        state = []  # List to keep track of history
        input_word = random.choice(vocab)
        cumulative_reward = 0
        
        for step in range(max_steps):
            # Convert the input word to its index in the vocabulary
            input_word_index = vocab.index(input_word)
            
            # Convert the index to a tensor
            input_tensor = torch.tensor([[input_word_index]], dtype=torch.long)
            
            # Predict the probabilities for the next word
            action_probs = model(input_tensor)
            
            # Select an action (word) based on the probabilities
            action_index = torch.multinomial(action_probs, 1).item()
            action_word = vocab[action_index]

            print(action_word)
            
            # Calculate the reward (similarity score)
            reward = similarity_function(target_word, action_word)
            
            # Update the state with the chosen action and reward
            state.append((action_word, reward))
            
            # Calculate cumulative reward
            cumulative_reward += reward
            
            # Compute loss and update the model
            loss = -torch.log(action_probs[0, action_index]) * reward
            
            optimizer.zero_grad()  # Clear the gradients
            loss.backward()  # Calculate gradients
            optimizer.step()  # Update model parameters
            
            # Update the input word for the next step
            input_word = action_word
            
            # End the episode if the reward is high enough (e.g., similarity close to 1)
            if reward >= 0.9:
                print(f"Episode {episode + 1}: Target word guessed correctly - '{target_word}'")
                break
        
        print(f"Episode {episode + 1}: Cumulative reward: {cumulative_reward}")

# Example usage
vocab = list(word2vec.keys()) #['apple', 'orange', 'banana', 'grape', 'cherry']  # Vocabulary list
vocab_size = len(vocab)
print(vocab_size)
embedding_dim = 50  # Size of the word embedding
hidden_dim = 64  # LSTM hidden state dimension

# Create the LSTM policy network
model = LSTMPolicyNetwork(vocab_size, embedding_dim, hidden_dim)

# Train the policy network with reinforcement learning
target_word = 'reluctant'  # Example target word
episodes = 400  # Number of episodes to train
max_steps = 400 # Maximum steps per episode

train_lstm_rl_policy(target_word, vocab, model, episodes, max_steps)

print(vocab_size)