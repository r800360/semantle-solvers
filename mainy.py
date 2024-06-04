import sqlite3
from numpy.linalg import norm
import numpy as np
import random
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR  # Import the learning rate scheduler
from torchinfo import summary

#from models.feedfoward import FeedForwardPolicyNetwork
from models.lstm import LSTMPolicyNetwork

class LSTMPolicyNetwork(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, batch_size):
        super(LSTMPolicyNetwork, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc_policy = nn.Linear(hidden_dim, vocab_size)
        self.fc_value = nn.Linear(hidden_dim, 1)
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.zeros(1, self.batch_size, self.hidden_dim),
                torch.zeros(1, self.batch_size, self.hidden_dim))

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        policy = self.fc_policy(lstm_out[:, -1, :])
        value = self.fc_value(lstm_out[:, -1, :])
        return torch.softmax(policy, dim=-1), value


cos_sim = nn.CosineSimilarity(dim=1)
word2vec = {}
for letter_range in ("a-c", "d-h", "i-o", "p-r", "s-z"):
    with sqlite3.connect(f"./data/word2vec_{letter_range}.db") as con:
        cur = con.execute("SELECT * FROM word2vec")
        for word, vec in cur:
            vec = np.frombuffer(vec, dtype=np.float32)
            word2vec[word] = vec / norm(vec)


def similarity_function(target_list, guess_list):
    # Extract vectors for the first word and the guess list
    x = torch.tensor([word2vec[g] for g in target_list])
    y = torch.tensor([word2vec[g] for g in guess_list])
    
    similarities = cos_sim(x,y)
    
    # Squaring curve mapping from [-1, 1] to [-1, 1]
    similarities = (similarities + 1) / 2
    similarities = similarities ** 0.5
    similarities = 2 * similarities - 1
    
    return similarities



def update_policy(rewards, log_probs, optimizer):
    log_probs = torch.stack(log_probs)
    # print("Log Probs")
    # print(log_probs)
    loss = -torch.mean(log_probs * sum(rewards))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss

def ppo_loss(old_log_probs, log_probs, advantages, epsilon=0.2):
    ratio = torch.exp(log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
    return policy_loss


def train_ppo_policy(vocab, model, episodes, max_steps, batch_size, epochs=10, epsilon=0.2):
    optimizer = optim.AdamW(model.parameters(), lr=0.005)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.1)

    episode_losses = []

    for episode in range(episodes):
        state = []
        input_words = random.choices(vocab, k=batch_size)
        target_words = random.choices(vocab, k=batch_size)

        print(f"Episode {episode + 1}: Input words: '{input_words}'")
        print(f"Episode {episode + 1}: Target words: '{target_words}'")

        log_probs = []
        rewards = []
        values = []

        for step in range(max_steps):
            input_word_indices = np.array([np.argmax(vocab == word) for word in input_words])
            input_tensor = torch.tensor(input_word_indices, dtype=torch.long).unsqueeze(1)
            action_probs, value = model(input_tensor)

            action_indices = torch.multinomial(action_probs, 1).squeeze()
            action_words = vocab[action_indices]
            reward = similarity_function(target_words, action_words)

            state.append((input_tensor, action_indices, reward))
            rewards.append(reward)
            values.append(value.squeeze())
            log_probs.append(torch.log(action_probs.gather(1, action_indices.unsqueeze(1)).squeeze()))

            input_words = action_words

            if (reward >= 0.9).any():
                break

        # Compute returns and advantages
        rewards = torch.tensor(rewards)
        values = torch.tensor(values)
        returns = rewards + 0.99 * values[1:].detach() - values[:-1]
        advantages = (returns - values[:-1]).detach()

        old_log_probs = torch.stack(log_probs)

        for _ in range(epochs):
            for input_tensor, action_indices, reward in state:
                action_probs, value = model(input_tensor)
                log_prob = torch.log(action_probs.gather(1, action_indices.unsqueeze(1)).squeeze())

                policy_loss = ppo_loss(old_log_probs, log_prob, advantages, epsilon)
                value_loss = (reward - value).pow(2).mean()

                loss = policy_loss + 0.5 * value_loss - 0.01 * log_prob.mean()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()

        scheduler.step()
        print(f"Episode {episode + 1}: Cumulative reward: {rewards.sum()}")
        episode_losses.append(loss.item())

        if isinstance(model, LSTMPolicyNetwork):
            model.reset_hidden()
        else:
            model.reset_parameters()

    print("Training complete")
    print("Episode losses:", episode_losses)

# def word_getter(vocab, desired_word):
#     #return index of desired word in vocab
#     for 

def main():
    vocab = ["apple", "car"]
    vocab = np.asarray(vocab)
    vocab_size = len(vocab)
    embedding_dim = 30
    hidden_dim = 100
    episodes = 100
    max_steps = 50
    batch_size = 10

    model = LSTMPolicyNetwork(vocab_size, embedding_dim, hidden_dim, batch_size)
    train_ppo_policy(vocab, model, episodes, max_steps, batch_size)

if __name__ == "__main__":
    main()



if __name__ == "__main__":
    main()