from src.models.lstm import LSTMPolicyNetwork
from src.train import TrainingOutcome
from torch import nn

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
    
    # Check if the model is an LSTM model
    if isinstance(model, LSTMPolicyNetwork):
        # Plot the hidden state
        hidden_state = training_outcome.hidden_state_samples[-1][0][-1]
        # print(hidden_state.shape) # (n,)
        
        # Plot the hidden state
        plt.figure()
        plt.plot(hidden_state)
        plt.title('Hidden State')
        plt.xlabel('Hidden Dimension')
        plt.ylabel('Hidden State Value')
        plt.savefig('artifacts/plots/hidden_state.png')