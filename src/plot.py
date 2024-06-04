import torch
from src.models.binary_ff import BinaryFeedForwardPolicyNetwork
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
    
    plot_loss_reward(training_outcome)
    
    ave_accuracies = training_outcome.episode_accuracy.mean(dim=1)
    plot_ave_accuracies(ave_accuracies)

    last_episode = training_outcome.episode_accuracy[-1]
    plot_last_episode_accuracy(last_episode)
    
    if isinstance(model, LSTMPolicyNetwork):
        hidden_states = training_outcome.hidden_state_samples
        
        # Enhanced Plots
        plot_hidden_state_evolution(hidden_states)
        plot_hidden_state_correlation(hidden_states[0, :, 0, -1, :], output_file='hidden_state_correlation_first_episode.png')
        plot_hidden_state_correlation(hidden_states[-1, :, 0, -1, :], output_file='hidden_state_correlation_last_episode.png')

        hidden_state = hidden_states[0, -1, 0, -1, :]  # First episode, last time step, first batch, last layer
        #plot_hidden_state_pca(hidden_state)
        #plot_hidden_state_tsne(hidden_state)
    
    if isinstance(model, BinaryFeedForwardPolicyNetwork):
        
        # Generate plot showing edge weights
        edge_weights = model.layers[0].weight.detach().numpy()
        plt.figure(figsize=(8, 6))
        sns.heatmap(edge_weights, cmap='viridis', cbar=True)
        plt.title('Edge Weights of the First Layer')
        plt.xlabel('Input Dimensions')
        plt.ylabel('Hidden Dimensions')
        plt.savefig('artifacts/plots/edge_weights.png')

        # Use torchviz to generate a visualization of the model
        from torchviz import make_dot
        e1 = torch.randint(0, 2, (10,)).float()
        e2 = torch.rand(10)

        print(e1, e2)

        dot = make_dot(model(e1, e2), params=dict(model.named_parameters()))
        dot.render('artifacts/plots/binary_ff_model', format='png')


def plot_loss_reward(training_outcome):
    # Split into multiple plots
    fig, axs = plt.subplots(2)
    fig.suptitle('Training Outcome')
    axs[0].plot(training_outcome.episode_losses, label='Loss')
    axs[0].set(ylabel='Loss')
    axs[1].plot(training_outcome.episode_rewards, label='Reward')
    axs[1].set(xlabel='Episode', ylabel='Reward')
    plt.savefig('artifacts/plots/training_outcome.png')

def plot_ave_accuracies(ave_accuracies):
    plt.figure(figsize=(12, 6))
    plt.plot(ave_accuracies)
    plt.title('Average Accuracy Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Average Accuracy')
    plt.savefig('artifacts/plots/average_accuracy.png')

def plot_last_episode_accuracy(last_episode):
    plt.figure(figsize=(12, 6))
    plt.plot(last_episode)
    plt.title('Accuracy in the Last Episode')
    plt.xlabel('Time Steps')
    plt.ylabel('Accuracy')
    plt.savefig('artifacts/plots/last_episode_accuracy.png')

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