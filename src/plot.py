from src.train import TrainingOutcome


def plot_data(training_outcome: TrainingOutcome):
    import matplotlib.pyplot as plt

    # Plot episode losses and rewards
    plt.figure(figsize=(10, 8))
    plt.plot(training_outcome.episode_losses, label='Loss')
    plt.plot(training_outcome.episode_rewards, label='Reward')
    plt.xlabel('Episode')
    plt.ylabel('Value')
    plt.title('Training Outcome')
    plt.legend()
    plt.grid(True)
    plt.savefig('artifacts/plots/training_outcome.png')