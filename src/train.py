import logging
import random

import numpy as np
#import spinup
from src.models.ppo_alg import ppo
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from src.models.lstm import LSTMPolicyNetwork
from src.similarity import similarity_function, similarity_to_reward



from src.models.actor_critic import ActorCritic
from src.models.semantle_env import SemantleEnv

logger = logging.getLogger(__name__)
    

class TrainingOutcome:
    def __init__(self, episode_losses=[], episode_rewards=[], episode_reward_differences=[]):
        self.episode_losses = episode_losses
        self.episode_rewards = episode_rewards
        self.episode_reward_differences = episode_reward_differences
        self.hidden_state_samples = torch.tensor([])
        self.episode_accuracy = torch.tensor([])

def flatten_dict(my_dict):
    return [item for pair in my_dict.items() for item in pair]

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

# def train_rl_policy_ppo(vocab, model, episodes, max_steps, batch_size, device: torch.device, args):
#     # Define input and output dimensions for the model
    
#     input_dim = 1  # Observation space size (e.g., similarity score)
#     output_dim = len(vocab)  # Action space size (e.g., vocabulary size)
    
#     # Create the ActorCritic model
#     ac_kwargs = {'input_dim': input_dim, 'output_dim': output_dim}
    
#     def create_env():
#         return SemantleEnv("orange", vocab)
    
#     # Run PPO
#     ac = ppo(env_fn=create_env, 
#                         actor_critic=ActorCritic, 
#                         ac_kwargs=ac_kwargs,
#                         seed=0, 
#                         steps_per_epoch=4000, 
#                         epochs=50)


def train_rl_policy(vocab, model, episodes, max_steps, batch_size, device: torch.device, args):
    gamma = 0.99
    previous_rewards = 0
    n = len(vocab)
    optimizer = optim.NAdam(model.parameters(), lr=0.1)
    # optimizer = optim.AdamW(model.parameters(), lr = 0.2)
    
    # Epsilon-greedy parameters
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.01
    
    # Track the total loss for each episode
    training_outcome = TrainingOutcome()

    for episode in range(episodes):
        # scheduler.step()  # Update the learning rate at the beginning of each episode

        # Initialize the state (history of words and similarity scores)
        states = {1: 0, 2: 0, 3:0}  # List to keep track of history
        states_list = []
        # input_word = random.choices(vocab)[0]#, k=batch_size)
        # target_word = random.choices(vocab)[0]#, k=batch_size)
        input_word = random.choices(vocab)[0]#, k=batch_size)vocab[0]
        target_word = random.choices(vocab)[0]#vocab[2]

        # if args.target_zero:
        #     target_words = [vocab[0]] #* batch_size

        logger.debug(f"Episode {episode + 1}: Input words: {input_word}")
        logger.debug(f"Episode {episode + 1}: Target words: {target_word}")


        log_probs = []
        rewards = []

        success = False
        
        for step in range(max_steps):
            # Convert the input word to its index in the vocabulary

            input_word_index = np.argmax(vocab == input_word) + 1
            
            # one_hot_input_word_index = np.zeros(n+1)
            # one_hot_input_word_index[input_word_index] = 1
            similarity = similarity_function([target_word], [input_word])[0]
            # one_hot_input_word_index[-1] = similarity
            # if (similarity < 0.99):
            #     similarity = 0
            # else:
            #     similarity = 1
             
            state = (input_word_index, similarity)
            
            if states[input_word_index] == 0:
                states[input_word_index] = similarity
            
            states_list.append(state)
                
            state_tensor = torch.FloatTensor(np.array(flatten_dict(states))).to(device)
            logger.debug(f"Similarity: {similarity}")
            
            # action, log_prob = model.get_action(state_tensor)
            if random.uniform(0, 1) < epsilon:
                action = random.choice(range(len(vocab)))
                log_prob = torch.log(torch.tensor(1.0 / len(vocab), requires_grad=True))
            else:
                action, log_prob = model.get_action(state_tensor)
            
            log_probs.append(log_prob)
            action_word = vocab[action]
            # print(action_word)
            
            # reward = similarity_to_reward(similarity, args).item()#[0]
            # reward = similarity-0.8 
            if input_word == target_word:
                reward = 1
            else:
                reward = -1
                # Reward proportional to similarity
                # reward = similarity - 0.5  # Encourage guesses closer to target

                # # Penalize guesses far from target
                # if similarity < 0.3:
                #     reward -= 0.5

                # # Time penalty
                # reward -= step * 10

                # Additional reward/penalty logic
                # ...
            
            rewards.append(reward)

            # if action_word == target_word:
            #     break
            
            logger.debug(f"Input word: {input_word}")
            logger.debug(f"Rewards: {reward}")
            input_word = action_word
            
            
        #returns = rewards_to_go(rewards)#compute_returns(rewards)
        returns = compute_returns(rewards)#srewards_to_go(rewards)
        
        
        # returns = torch.tensor(returns).to(device)

        # # Normalize rewards
        # returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        # # returns = torch.tensor(returns)
        policy_gradient = []
        for log_prob, R in zip(log_probs, returns):
            policy_gradient.append(-log_prob * R  )
        optimizer.zero_grad()
        
        policy_gradient = torch.stack(policy_gradient).sum()
        policy_gradient.backward()
        optimizer.step()
        
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
        
        # Print the cumulative reward and average loss for the episode
        logger.info(f"Episode {episode + 1}: Cumulative reward: {sum(rewards)}")
        logger.info(f"Episode {episode + 1}: Loss: {policy_gradient}")
        
        training_outcome.episode_losses.append(policy_gradient.detach().numpy().item())
        training_outcome.episode_rewards.append(sum(rewards))
        
    logger.info("Training complete")
    logger.info("Episode losses: " + str(training_outcome.episode_losses))
    
    return training_outcome