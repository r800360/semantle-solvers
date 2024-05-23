from src.train import TrainingOutcome


def plot(training_outcome: TrainingOutcome):
    import matplotlib.pyplot as plt

    # Plot the training outcome
    plt.figure(figsize=(12, 6))
    plt.plot(training_outcome)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Outcome')
    plt.grid()
    
    # Store to artifacts/plots
    plt.savefig('artifacts/plots/training_outcome.png')