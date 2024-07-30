import logging
import random

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from src.models.lstm import LSTMPolicyNetwork
from src.similarity import similarity_function, similarity_to_reward

logger = logging.getLogger(__name__)
    

class TrainingOutcome:
    def __init__(self, episode_losses=[], episode_rewards=[], episode_reward_differences=[]):
        self.episode_losses = episode_losses
        self.episode_rewards = episode_rewards
        self.episode_reward_differences = episode_reward_differences
        self.hidden_state_samples = torch.tensor([])
        self.episode_accuracy = torch.tensor([])

def compute_returns(rewards, gamma=0.99):
    R = 0
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns

def rewards_to_go(rewards, gamma=0.99):
    n = len(rewards)
    rtgs = np.zeros_like(rewards)
    for i in reversed(range(n)):
        rtgs[i] = rewards[i] + (rtgs[i+1] if i+1 < n else 0)
    return rtgs


def train_rl_policy(vocab, model, episodes, max_steps, batch_size, device: torch.device, args):
    gamma = 0.99
    previous_rewards = 0
    optimizer = optim.NAdam(model.parameters(), lr=0.01)
    # optimizer = optim.AdamW(model.parameters(), lr = 0.2)

    # Track the total loss for each episode
    training_outcome = TrainingOutcome()

    for episode in range(episodes):
        # scheduler.step()  # Update the learning rate at the beginning of each episode

        # Initialize the state (history of words and similarity scores)
        states = []  # List to keep track of history
        input_word = "orange" #* batch_size #random.choices(vocab, k=batch_size)
        target_word = "orange" #* batch_size #random.choices(vocab, k=batch_size)

        # if args.target_zero:
        #     target_words = [vocab[0]] #* batch_size

        logger.debug(f"Episode {episode + 1}: Input words: {input_word}")
        logger.debug(f"Episode {episode + 1}: Target words: {target_word}")


        log_probs = []
        rewards = []

        success = False
        
        for step in range(max_steps):
            # Convert the input word to its index in the vocabulary

            # input_word_indices = np.array([np.argmax(vocab == word) for word in input_words])
            # input_tensor = torch.tensor(input_word_indices, dtype=torch.long).to(device)
            input_word_index = np.argmax(vocab == input_word)
            # input_tensor = torch.tensor([input_word_index], dtype=torch.long).to(device)
            
            similarity = similarity_function([target_word], [input_word])[0]
            state = (input_word_index, similarity)
            states.append(state)
            state_tensor = torch.FloatTensor(state).to(device)
            
            logger.debug(f"Similarity: {similarity}")
            
            action, log_prob = model.get_action(state_tensor)
            log_probs.append(log_prob)
            action_word = vocab[action]
            # print(action_word)
            
            # reward = similarity_to_reward(similarity, args).item()#[0]
            
            if input_word == target_word:
                reward = 1
            else:
                reward = -1
            
            rewards.append(reward)

            # if action_word == target_word:
            #     break
            
            logger.debug(f"Input word: {input_word}")
            logger.debug(f"Rewards: {reward}")
            input_word = action_word
            # state_tensor = torch.tensor(state, dtype=torch.float).to(device)

            # Unsqueeze to make batch of len 1 sequences
            # input_tensor = input_tensor.unsqueeze(1)

            # Predict the probabilities for the next word
            # And back to cpu land
            # similarity = similarity_function(target_words, input_words)
            # action_probs = model(input_tensor, similarity).squeeze().cpu()

            # action_indices = torch.multinomial(action_probs, 1).squeeze()
            # action_words = [vocab[idx] for idx in action_indices]#vocab[action_indices]

            # Compute loss using REINFORCE algorithm
            # chosen_action_probs = action_probs.gather(1, action_indices.unsqueeze(1)).squeeze()
            # logger.debug(chosen_action_probs)
            # log_prob = torch.log(chosen_action_probs)
            # log_probs = torch.cat((log_probs, log_prob.unsqueeze(0)))

            
            # Update the input word for the next step
            # input_words = action_words
            
        # If success, reward is 1. Otherwise, -1
        # succ_rewards = torch.tensor([1 if s else -1 for s in success], dtype=torch.float)
        #returns = rewards_to_go(rewards)#compute_returns(rewards)
        returns = compute_returns(rewards)
        # returns = torch.tensor(returns)
        policy_gradient = []
        for log_prob, R in zip(log_probs, returns):
            policy_gradient.append(-log_prob * R  )
        optimizer.zero_grad()
        
        policy_gradient = torch.stack(policy_gradient).sum()
        policy_gradient.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        # log_probs = torch.stack(log_probs)
        # Update the policies
        #print(log_probs)

        # loss = -torch.mean(log_probs * sum(rewards))
        
        # if epsilon > epsilon_min:
        #     epsilon *= epsilon_decay

        # optimizer.zero_grad()
        # loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        # optimizer.step()
        # scheduler.step()  # Update the learning rate at the end of each episode
        
        # logger.debug(f"Log Probs: {log_probs}")
        
        # Save the accuracy for the episode
        # training_outcome.episode_accuracy = torch.cat((training_outcome.episode_accuracy, accuracies.unsqueeze(0)), dim=0)

        # Print the cumulative reward and average loss for the episode
        logger.info(f"Episode {episode + 1}: Cumulative reward: {sum(rewards)}")
        logger.info(f"Episode {episode + 1}: Loss: {policy_gradient}")
        
        training_outcome.episode_losses.append(policy_gradient.detach().numpy().item())
        training_outcome.episode_rewards.append(sum(rewards))
        
    logger.info("Training complete")
    logger.info("Episode losses: " + str(training_outcome.episode_losses))
    
    return training_outcome