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
    
    # Sample usage:
    if isinstance(model, LSTMPolicyNetwork):
        hidden_states = training_outcome.hidden_state_samples
        
        hidden_state = hidden_states[0, -1, 0, -1, :]  # First episode, last time step, first batch, last layer

        # Original Plot
        plt.figure()
        plt.plot(hidden_state)
        plt.title('Hidden State')
        plt.xlabel('Hidden Dimension')
        plt.ylabel('Hidden State Value')
        plt.savefig('artifacts/plots/hidden_state.png')
        plt.show()
        
        # Enhanced Plots
        plot_hidden_state_evolution(hidden_states)
        plot_hidden_state_correlation(hidden_state)
        plot_hidden_state_pca(hidden_state)
        plot_hidden_state_tsne(hidden_state)

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

def plot_hidden_state_correlation(hidden_state):
    correlation_matrix = np.corrcoef(hidden_state.T)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, cmap='coolwarm', cbar=True, annot=True, fmt=".2f")
    plt.title('Hidden State Correlation Matrix')
    plt.xlabel('Hidden Dimensions')
    plt.ylabel('Hidden Dimensions')
    plt.savefig('artifacts/plots/hidden_state_correlation.png')

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