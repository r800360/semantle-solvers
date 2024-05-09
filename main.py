import sqlite3
from numpy.linalg import norm
import numpy as np
import random
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR  # Import the learning rate scheduler

from models.feedfoward import FeedForwardPolicyNetwork
from models.lstm import LSTMPolicyNetwork

word2vec = {}
for letter_range in ("a-c", "d-h", "i-o", "p-r", "s-z"):
    with sqlite3.connect(f"./data/word2vec_{letter_range}.db") as con:
        cur = con.execute("SELECT * FROM word2vec")
        for word, vec in cur:
            vec = np.frombuffer(vec, dtype=np.float32)
            word2vec[word] = vec / norm(vec)


def similarity_function(first_word, guess_list):
    # Extract vectors for the first word and the guess list
    x = word2vec[first_word]
    y = np.array([word2vec[g] for g in guess_list])
    
    # Calculate the cosine similarity using dot product and norms
    # Using broadcasting for efficient computation
    x_norm = np.linalg.norm(x)
    y_norms = np.linalg.norm(y, axis=1)
    similarities = np.dot(y, x) / (x_norm * y_norms)
    
    # Apply transformation to change range from [-1, 1] to [0, 1]
    similarities = (similarities + 1) / 2
    
    # Apply a squaring curve to emphasize larger values
    similarities = similarities ** 2
    
    # Transform back to range [-1, 1]
    similarities = 2 * similarities - 1
    
    return similarities



def update_policy(rewards, log_probs, optimizer):
    log_probs = torch.stack(log_probs)
    print(log_probs)
    loss = -torch.mean(log_probs * sum(rewards))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss

# Reinforcement learning training loop
def train_rl_policy(target_word, vocab, model, episodes, max_steps, batch_size):
    optimizer = optim.AdamW(model.parameters(), maximize=False, lr=0.01)
    
    # Define the learning rate scheduler
    # scheduler = StepLR(optimizer, step_size=100, gamma=0.1)  # Adjust step_size and gamma as needed
    
    # Track the total loss for each episode
    episode_losses = []
    
    for episode in range(episodes):
        #scheduler.step()  # Update the learning rate at the beginning of each episode
        
        # Initialize the state (history of words and similarity scores)
        state = []  # List to keep track of history
        input_words = random.choices(vocab, k=batch_size)
        target_words = random.choices(vocab, k=batch_size)
        print(f"Episode {episode + 1}: Target word: '{target_word}'")

        log_probs = []
        all_rewards = []
        
        for step in range(max_steps):
            # Convert the input word to its index in the vocabulary
            # input_word_index = vocab.index(input_word)
            input_word_indices = np.array([np.argmax(vocab == word) for word in input_words])#input_word_indices = [np.where(vocab == input_word) for input_word in input_words]
            
            # Convert the index to a tensor
            input_tensor = torch.tensor(input_word_indices, dtype=torch.long)
            
            print(input_tensor)
            
            # Predict the probabilities for the next word
            action_probs = model(input_tensor)
            
            # Select an action (word) based on the probabilities
            # action_index = torch.multinomial(action_probs, 1).item()
            action_indices = torch.multinomial(action_probs, 1).squeeze()
            print(action_indices)
            
            
            action_words = vocab[action_indices]

            print(action_words)
            
            # Calculate the reward (similarity score)
            rewards = similarity_function(target_word, action_words)
            
            # Update the state with the chosen action and reward
            state.append((action_words, rewards))
            
            # Calculate cumulative reward
            all_rewards = np.concatenate((all_rewards,rewards), axis = 0)
        
            # Compute loss using REINFORCE algorithm
            chosen_action_probs = action_probs.gather(1, action_indices.unsqueeze(1)).squeeze()
            log_prob = torch.log(chosen_action_probs)
            # log_prob = torch.log(torch.tensor([probs[action_indices] for probs in action_probs]))
            
            log_probs.append(log_prob)
            
            # Update the model parameters using the policy gradient
            optimizer.zero_grad()  # Clear the gradients
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()  # Update model parameters
            
            # Update the input word for the next step
            input_words = action_words
            
            # End the episode if the reward is high enough (e.g., similarity close to 1)
            for reward, target_word in zip(rewards, target_words):
                if reward >= 0.9:
                    print(f"Episode {episode + 1}: Target word guessed correctly - '{target_word}'")
                    break

        # Update the policy
        loss = update_policy(all_rewards, log_probs, optimizer)
        episode_losses.append(loss.item())

        # Print the cumulative reward and average loss for the episode
        print(f"Episode {episode + 1}: Cumulative reward: {sum(all_rewards)}")
        if len(episode_losses) > 0:
            # avg_loss = sum(episode_losses) / len(episode_losses)
            print(f"Episode {episode + 1}: Average loss: {loss}")
        
        # Clear the list of episode losses for the next episode
        episode_losses = []

def main():
    # Example usage
    # vocab = ["apple", "banana", "orange", "grape", "mango", "pineapple", "strawberry", "blueberry", "raspberry", "watermelon", "kiwi", "pear", "peach", "plum", "cherry", "lemon", "lime", "papaya", "guava", "avocado", "cranberry", "grapefruit", "coconut", "lychee", "passionfruit", "fig", "date", "pomegranate", "cantaloupe", "nectarine", "apricot", "persimmon", "tangerine", "clementine", "dragonfruit", "starfruit", "blackberry", "elderberry", "jackfruit"]
    # vocab = ["apple", "banana", "orange", "grape", "mango"]
    vocab = list(word2vec.keys())
    vocab = vocab[:100]

    vocab = np.asarray(vocab)
    #vocab = torch.from_numpy(vocab)
    #vocab = torch.tensor(vocab)

    vocab_size = len(vocab)
    target_word = vocab[26]
    # target_word = 'apple'  # Example target word
    print("Vocab size: " + str(vocab_size))
    print(f"Target word: '{target_word}'")
    
    embedding_dim = 30  # Size of the word embedding
    hidden_dim = 100  # LSTM hidden state dimension

    # Create the LSTM policy network
    # model = FeedForwardPolicyNetwork(vocab_size, embedding_dim, hidden_dim)
    model = LSTMPolicyNetwork(vocab_size, embedding_dim, hidden_dim)
    
    episodes = 20  # Number of episodes to train
    max_steps = 50 # Maximum steps per episode
    batch_size = 10

    train_rl_policy(target_word, vocab, model, episodes, max_steps, batch_size)

    print(vocab_size)

if __name__ == "__main__":
    main()