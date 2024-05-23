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
    similarities = similarities ** 0.8
    similarities = 2 * similarities - 1
    
    return similarities

class TrainingOutcome:
    def __init__(self, episode_losses=[], episode_rewards=[], episode_reward_differences=[]):
        self.episode_losses = episode_losses
        self.episode_rewards = episode_rewards
        self.episode_reward_differences = episode_reward_differences
        self.hidden_state_samples = torch.tensor([])

def train_rl_policy(vocab, model, episodes, max_steps, batch_size, device: torch.device):
    previous_rewards = 0
    optimizer = optim.AdamW(model.parameters(), maximize=False, lr=0.005)

    # Define the learning rate scheduler
    scheduler = StepLR(optimizer, step_size=100, gamma=0.1)  # Adjust step_size and gamma as needed

    # Track the total loss for each episode
    training_outcome = TrainingOutcome()

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
        all_reward_differences = []
        hidden_states = torch.tensor([])

        for step in range(max_steps):
            # Convert the input word to its index in the vocabulary

            input_word_indices = np.array([np.argmax(vocab == word) for word in input_words])
            input_tensor = torch.tensor(input_word_indices, dtype=torch.long).to(device)

            # Unsqueeze to make batch of len 1 sequences
            input_tensor = input_tensor.unsqueeze(1)

            # Predict the probabilities for the next word
            # And back to cpu land
            action_probs = model(input_tensor).squeeze().cpu()

            action_indices = torch.multinomial(action_probs, 1).squeeze()
            action_words = vocab[action_indices]

            # Calculate the reward (similarity score)
            if (step > 1): 
                previous_rewards = rewards
            reward_scaler = (step/max_steps)**0.5
            rewards = similarity_function(target_words, action_words) * reward_scaler
            rewards_difference = rewards - previous_rewards
            
            # Update the state with the chosen action and reward
            state.append((action_words, rewards))

            # Calculate cumulative reward
            all_rewards = np.concatenate((all_rewards,rewards), axis = 0)
            if (step > 1):
                all_reward_differences = np.concatenate((all_reward_differences, rewards_difference), axis = 0)


            # Compute loss using REINFORCE algorithm
            chosen_action_probs = action_probs.gather(1, action_indices.unsqueeze(1)).squeeze()
            log_prob = torch.log(chosen_action_probs)

            log_probs.append(log_prob)

            #Gradient Clipping
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)

            # Update the input word for the next step
            input_words = action_words
            
            # Update hidden states
            if isinstance(model, LSTMPolicyNetwork):
                hidden_states = torch.cat((hidden_states, model.hidden_state.detach().cpu().unsqueeze(0)), dim=0)

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

        # Update the policy
        log_probs = torch.stack(log_probs)
        loss = -torch.mean(log_probs * sum(all_reward_differences))
        optimizer.zero_grad()
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        scheduler.step()  # Update the learning rate at the end of each episode
        
        logger.debug(f"Log Probs: {log_probs}")
        
        
        # Save the hidden state for the model
        # Shaped: Episode x step x batch x layer x hidden_dim
        if isinstance(model, LSTMPolicyNetwork):
            training_outcome.hidden_state_samples = torch.cat((training_outcome.hidden_state_samples, hidden_states.unsqueeze(0)), dim=0)
        
        

        # Reset the model parameters for the next episode
        if isinstance(model, LSTMPolicyNetwork):
            model.reset_hidden(device)

        # Print the cumulative reward and average loss for the episode
        logger.info(f"Episode {episode + 1}: Cumulative reward: {sum(all_rewards)}")
        logger.info(f"Episode {episode + 1}: Reward Differences: {sum(all_reward_differences)}")
        logger.info(f"Episode {episode + 1}: Average loss: {loss}")
        
        training_outcome.episode_losses.append(loss.detach().numpy())
        training_outcome.episode_rewards.append(sum(all_rewards))
        training_outcome.episode_reward_differences.append(sum(all_reward_differences))

    logger.info("Training complete")
    logger.info("Episode losses: " + str(training_outcome.episode_losses))
    
    return training_outcome