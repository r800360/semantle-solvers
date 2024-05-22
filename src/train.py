import logging
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from src.models.lstm import LSTMPolicyNetwork
from src.data import word2vec

logger = logging.getLogger(__name__)

cos_sim = nn.CosineSimilarity(dim=1)

def similarity_function(target_list, guess_list):
    # Extract vectors for the first word and the guess list
    x = torch.tensor(np.array([word2vec[g] for g in target_list]))
    y = torch.tensor(np.array([word2vec[g] for g in guess_list]))
    
    similarities = cos_sim(x,y)
    
    # Squaring curve mapping from [-1, 1] to [-1, 1]
    similarities = (similarities + 1) / 2
    similarities = similarities ** 0.5
    similarities = 2 * similarities - 1
    
    return similarities



def update_policy(rewards, log_probs, optimizer):
    log_probs = torch.stack(log_probs)
    loss = -torch.mean(log_probs * sum(rewards))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    logger.debug(f"Log Probs: {log_probs}")
    return loss

def train_rl_policy(vocab, model, episodes, max_steps, batch_size, device: torch.device):
    optimizer = optim.AdamW(model.parameters(), maximize=True, lr=0.005)

    # Define the learning rate scheduler
    scheduler = StepLR(optimizer, step_size=100, gamma=0.1)  # Adjust step_size and gamma as needed

    # Define the learning rate scheduler
    # scheduler = StepLR(optimizer, step_size=100, gamma=0.1)  # Adjust step_size and gamma as needed

    # Track the total loss for each episode
    episode_losses = []

    # Reset the model parameters before starting the training loop
    # if isinstance(model, LSTMPolicyNetwork):
    #     model.reset_hidden()
    # else:
    #     model.reset_parameters()

    for episode in range(episodes):
        # scheduler.step()  # Update the learning rate at the beginning of each episode

        # Initialize the state (history of words and similarity scores)
        state = []  # List to keep track of history
        input_words = random.choices(vocab, k=batch_size)
        target_words = random.choices(vocab, k=batch_size)

        logger.debug(f"Episode {episode + 1}: Input words: {input_words}")
        logger.debug(f"Episode {episode + 1}: Target words: {target_words}")

        log_probs = []
        all_rewards = []

        for step in range(max_steps):
            # Convert the input word to its index in the vocabulary

            input_word_indices = np.array([np.argmax(vocab == word) for word in input_words])
            input_tensor = torch.tensor(input_word_indices, dtype=torch.long).to(device)

            # Unsqueeze to make batch of len 1 sequences
            input_tensor = input_tensor.unsqueeze(1)
            # print(input_word_indices)
            # print(input_tensor)
            # summary(model, input_data=input_tensor)
            # exit()

            # Predict the probabilities for the next word
            # And back to cpu land
            action_probs = model(input_tensor).squeeze().cpu()

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

            log_probs.append(log_prob)

            # Update the model parameters using the policy gradient
            optimizer.zero_grad()  # Clear the gradients
            # loss = -torch.mean(log_probs * sum(all_rewards))
            # loss.backward()

            #Gradient Clipping
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

            optimizer.step()  # Update model parameters

            # Update the input word for the next step
            input_words = action_words

            logger.debug(f"Input Word Indices: {input_word_indices}")
            logger.debug(f"Input Tensor: {input_tensor}")
            logger.debug(f"Action Probs: {action_probs}")
            logger.debug(f"Action indices: {action_indices}")
            logger.debug(f"Action words: {action_words}")
            logger.debug(f"Rewards: {rewards}")

            # End the episode if the reward is high enough (e.g., similarity close to 1)
            correct_guesses = rewards >= 0.9
            if correct_guesses.any():
                logger.debug(f"Episode {episode + 1}: Guess correctness - {correct_guesses}")

        #scheduler.step()  # Update the learning rate at the end of each episode
        # Update the policy
        loss = update_policy(all_rewards, log_probs, optimizer)
        # episode_losses.append(loss.item())
        scheduler.step()
        # Reset the model parameters for the next episode
        if isinstance(model, LSTMPolicyNetwork):
            model.reset_hidden(device)

        # Print the cumulative reward and average loss for the episode
        logger.info(f"Episode {episode + 1}: Cumulative reward: {sum(all_rewards)}")
        if len(episode_losses) > 0:
            logger.info(f"Episode {episode + 1}: Average loss: {loss}")

    logger.info("Training complete")
    logger.info("Episode losses: " + str(episode_losses))