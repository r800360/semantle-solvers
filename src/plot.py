import torch
from src.models.lstm import LSTMPolicyNetwork
from src.train import TrainingOutcome
from torch import nn

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def plot_data(training_outcome: TrainingOutcome, model: nn.Module):
    import matplotlib.pyplot as plt
    
    # Split into multiple plots
    fig, axs = plt.subplots(3)
    fig.suptitle('Training Outcome')
    axs[0].plot(training_outcome.episode_losses, label='Loss')
    axs[0].set(ylabel='Loss')
    axs[1].plot(training_outcome.episode_rewards, label='Reward')
    axs[1].set(xlabel='Episode', ylabel='Reward')
    axs[2].plot(training_outcome.episode_reward_differences, label='Reward Difference')
    axs[2].set(xlabel='Episode', ylabel='Reward Difference')
    plt.savefig('artifacts/plots/training_outcome.png')
    
    # Plot average accuracies [Episode x Step]
    # 1. Calculate the average accuracy for each episode
    # 2. Plot the accuracy progression over the first and final episode
    
    ave_accuracies = torch.mean(training_outcome.episode_accuracy, dim=1)
    plt.figure(figsize=(12, 6))
    plt.plot(ave_accuracies)
    plt.title('Average Accuracy Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Average Accuracy')
    plt.savefig('artifacts/plots/average_accuracy.png')
    
    last_episode = training_outcome.episode_accuracy[-1, :]
    plt.figure(figsize=(12, 6))
    plt.plot(last_episode)
    plt.title('Accuracy in the Last Episode')
    plt.xlabel('Time Steps')
    plt.ylabel('Accuracy')
    plt.savefig('artifacts/plots/last_episode_accuracy.png')
    
    # Sample usage:
    if isinstance(model, LSTMPolicyNetwork):
        hidden_states = training_outcome.hidden_state_samples
        
        hidden_state = hidden_states[0, -1, 0, -1, :]  # First episode, last time step, first batch, last layer
        
        # Enhanced Plots
        plot_hidden_state_evolution(hidden_states)

        plot_hidden_state_correlation(hidden_states[0, :, 0, -1, :], output_file='hidden_state_correlation_first_episode.png')
        plot_hidden_state_correlation(hidden_states[-1, :, 0, -1, :], output_file='hidden_state_correlation_last_episode.png')
        
        #plot_hidden_state_pca(hidden_state)
        #plot_hidden_state_tsne(hidden_state)

def plot_hidden_state_evolution(hidden_states):
    # hidden_states shape: (episode, timestep, batch, layer, hidden_dim)
    # Extracting for the first episode, first batch, and the last layer
    hidden_state_evolution = hidden_states[0, :, 0, -1, :]
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(hidden_state_evolution.T, cmap='viridis', cbar=True)
    plt.title('Hidden State Evolution Over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Hidden Dimensions')
    plt.savefig('artifacts/plots/hidden_state_evolution.png')

def plot_hidden_state_correlation(hidden_state, output_file='hidden_state_correlation.png'):
    
    correlation_matrix = np.corrcoef(hidden_state.T)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, cmap='coolwarm', cbar=True, annot=False, fmt=".2f")
    plt.title('Hidden State Correlation Matrix')
    plt.xlabel('Hidden Dimensions')
    plt.ylabel('Hidden Dimensions')
    plt.savefig(f'artifacts/plots/{output_file}')

def plot_hidden_state_pca(hidden_state):
    pca = PCA(n_components=2)
    reduced_hidden_state = pca.fit_transform(hidden_state)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_hidden_state[:, 0], reduced_hidden_state[:, 1], alpha=0.7)
    plt.title('Hidden State PCA')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.savefig('artifacts/plots/hidden_state_pca.png')

def plot_hidden_state_tsne(hidden_state):
    tsne = TSNE(n_components=2, random_state=42)
    reduced_hidden_state = tsne.fit_transform(hidden_state)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_hidden_state[:, 0], reduced_hidden_state[:, 1], alpha=0.7)
    plt.title('Hidden State t-SNE')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.savefig('artifacts/plots/hidden_state_tsne.png')