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
    similarities = similarities ** 2
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

# Reinforcement learning training loop
def train_rl_policy(vocab, model, episodes, max_steps, batch_size):
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
        
        print(f"Episode {episode + 1}: Input words: '{input_words}'")
        print(f"Episode {episode + 1}: Target words: '{target_words}'")
        
        log_probs = []
        all_rewards = []
        
        for step in range(max_steps):
            # Convert the input word to its index in the vocabulary
            
            input_word_indices = np.array([np.argmax(vocab == word) for word in input_words])
            input_tensor = torch.tensor(input_word_indices, dtype=torch.long)
            
            # Unsqueeze to make batch of len 1 sequences
            input_tensor = input_tensor.unsqueeze(1)
            # print(input_word_indices)
            # print(input_tensor)
            
            # Predict the probabilities for the next word
            action_probs = model(input_tensor).squeeze()
            # Select an action (word) based on the probabilities
            # action_index = torch.multinomial(action_probs, 1).item()

            action_indices = torch.multinomial(action_probs, 1).squeeze()
            action_words = vocab[action_indices]
            
            # Calculate the reward (similarity score)
            rewards = similarity_function(target_words, action_words)
            
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
            
            # print("Input Word Indices " + str(input_word_indices))
            # print("Input Tensor " + str(input_tensor))
            # print("Action Probs " + str(action_probs))
            # print("Action indices " + str(action_indices))
            # print("Action words " + str(action_words))
            # print("Rewards " + str(rewards))
            
            # End the episode if the reward is high enough (e.g., similarity close to 1)
            correct_guesses = rewards >= 0.9
            if correct_guesses.any():
                print(f"Episode {episode + 1}: Guess correctness - {correct_guesses}")

        # Update the policy
        loss = update_policy(all_rewards, log_probs, optimizer)
        episode_losses.append(loss.item())
        
        # Reset the model parameters for the next episode
        if isinstance(model, LSTMPolicyNetwork):
            model.reset_hidden()
        else:
            model.reset_parameters()

        # Print the cumulative reward and average loss for the episode
        print(f"Episode {episode + 1}: Cumulative reward: {sum(all_rewards)}")
        if len(episode_losses) > 0:
            # avg_loss = sum(episode_losses) / len(episode_losses)
            print(f"Episode {episode + 1}: Average loss: {loss}")
        
    print("Training complete")
    print("Episode losses: " + str(episode_losses))

# def word_getter(vocab, desired_word):
#     #return index of desired word in vocab
#     for 

def main():
    # Example usage
    vocab = ["apple", "banana", "orange", "grape", "mango", "pineapple", "strawberry", "blueberry", "raspberry", "watermelon", "kiwi", "pear", "peach", "plum", "cherry", "lemon", "lime", "papaya", "guava", "avocado", "cranberry", "grapefruit", "coconut", "lychee", "passionfruit", "fig", "date", "pomegranate", "cantaloupe", "nectarine", "apricot", "persimmon", "tangerine", "clementine", "dragonfruit", "starfruit", "blackberry", "elderberry", "jackfruit"]
    # vocab = ["apple", "banana", "orange", "grape", "mango"]
    # vocab = list(word2vec.keys())
    # vocab = vocab[:100]

    vocab = np.asarray(vocab)
    vocab_size = len(vocab)
    
    embedding_dim = 30  # Size of the word embedding
    hidden_dim = 100  # LSTM hidden state dimension
    episodes = 100  # Number of episodes to train
    max_steps = 50 # Maximum steps per episode
    batch_size = 10

    # Create the LSTM policy network
    # model = FeedForwardPolicyNetwork(vocab_size, embedding_dim, hidden_dim)
    model = LSTMPolicyNetwork(vocab_size, embedding_dim, hidden_dim, batch_size)
    

    train_rl_policy(vocab, model, episodes, max_steps, batch_size)


if __name__ == "__main__":
    main()