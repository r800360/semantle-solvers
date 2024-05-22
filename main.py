import argparse
import logging
import sqlite3

import numpy as np
import torch
import torch.nn as nn

import src.models as models
from src.models.feedfoward import FeedForwardPolicyNetwork
from src.models.lstm import LSTMPolicyNetwork
from src.train import train_rl_policy

from src.data import load_w2vec

logger = logging.getLogger(__name__)

# Reinforcement learning training loop
def main(model_type: models.ModelType):
    
    # Load w2vec
    load_w2vec()
    
    # Example usage
    #vocab = ["apple", "banana", "orange", "grape", "mango", "pineapple", "strawberry", "blueberry", "raspberry", "watermelon", "kiwi", "pear", "peach", "plum", "cherry", "lemon", "lime", "papaya", "guava", "avocado", "cranberry", "grapefruit", "coconut", "lychee", "passionfruit", "fig", "date", "pomegranate", "cantaloupe", "nectarine", "apricot", "persimmon", "tangerine", "clementine", "dragonfruit", "starfruit", "blackberry", "elderberry", "jackfruit"]
    # vocab = ["apple", "banana", "orange", "grape", "mango"]
    vocab = ["apple", "car"]
    # exit(0)
    # vocab = list(word2vec.keys())
    # vocab = vocab[:100]

    vocab = np.asarray(vocab)
    vocab_size = len(vocab)
    
    embedding_dim = 30  # Size of the word embedding
    hidden_dim = 100  # LSTM hidden state dimension
    episodes = 100  # Number of episodes to train
    max_steps = 50 # Maximum steps per episode
    batch_size = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the LSTM policy network
    if model_type == models.ModelType.Feedforward:
        model = FeedForwardPolicyNetwork(vocab_size, embedding_dim, hidden_dim).to(device)
    elif model_type == models.ModelType.LSTM:
        model = LSTMPolicyNetwork(vocab_size, embedding_dim, hidden_dim, batch_size, device).to(device)
    else:
        raise ValueError("Invalid model type")

    train_rl_policy(vocab, model, episodes, max_steps, batch_size, device)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-d', '--debug', action='store_true',
                        help='Enable debug logging')
    parser.add_argument('-o', '--output', type=str, default='stdout', help='Output file')
    parser.add_argument('-m', '--model', type=models.ModelType, choices=list(models.ModelType), help='Model type (feedforward or lstm)')
    
    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    model_type = args.model
    
    output_file = args.output
    if output_file != 'stdout':
        logging.basicConfig(filename=f"logs/{output_file}", filemode='w', level=log_level)
    else:
        logging.basicConfig(level=log_level)
    
    main(model_type)