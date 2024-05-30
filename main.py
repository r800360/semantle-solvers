import argparse
import csv
import logging
import os
import sqlite3

import numpy as np
import torch
import torch.nn as nn

from src.plot import plot_data
import src.models as models
from src.data import load_data
from src.models.feedfoward import FeedForwardPolicyNetwork
from src.models.lstm import LSTMPolicyNetwork
from src.train import train_rl_policy

logger = logging.getLogger(__name__)

# Reinforcement learning training loop
def main(args):
    
    # Load w2vec
    load_data(args.dataset)
    
    # Example usage
    #vocab = ["apple", "banana", "orange", "grape", "mango", "pineapple", "strawberry", "blueberry", "raspberry", "watermelon", "kiwi", "pear", "peach", "plum", "cherry", "lemon", "lime", "papaya", "guava", "avocado", "cranberry", "grapefruit", "coconut", "lychee", "passionfruit", "fig", "date", "pomegranate", "cantaloupe", "nectarine", "apricot", "persimmon", "tangerine", "clementine", "dragonfruit", "starfruit", "blackberry", "elderberry", "jackfruit"]
    # vocab = ["apple", "banana", "orange", "grape", "mango"]
    # vocab = ["apple", "car"]
    # exit(0)
    # vocab = list(word2vec.keys())
    # vocab = vocab[:100]
    
    # Read vocab from vocab file
    if os.path.exists(args.vocab):
        with open(args.vocab, 'r') as f:
            # Read lines
            vocab = f.readlines()
            vocab = [v.strip() for v in vocab]
    else:
        raise ValueError("Vocabulary file does not exist")

    vocab = np.asarray(vocab)
    vocab_size = len(vocab)
    
    embedding_dim = 30  # Size of the word embedding
    hidden_dim = 100  # LSTM hidden state dimension
    episodes = args.episodes  # Number of episodes to train
    max_steps = 50 # Maximum steps per episode
    batch_size = args.batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the LSTM policy network
    if args.model == models.ModelType.Feedforward:
        model = FeedForwardPolicyNetwork(vocab_size, embedding_dim, hidden_dim).to(device)
    elif args.model == models.ModelType.LSTM:
        model = LSTMPolicyNetwork(vocab_size, embedding_dim, hidden_dim, batch_size, device).to(device)
    else:
        raise ValueError("Invalid model type")

    outcome = train_rl_policy(vocab, model, episodes, max_steps, batch_size, device)
    plot_data(outcome, model)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-d', '--debug', action='store_true',
                        help='Enable debug logging')
    parser.add_argument('-o', '--output', type=str, default='stdout', help='Output file')
    parser.add_argument('-m', '--model', type=models.ModelType, choices=list(models.ModelType), help='Model type (feedforward or lstm)', default=models.ModelType.LSTM)
    
    parser.add_argument('-e', '--episodes', type=int, default=100, help='Number of episodes')
    parser.add_argument('-b', '--batch_size', type=int, default=10, help='Batch size')
    parser.add_argument('-l', '--learning_rate', type=float, default=0.005, help='Learning rate')
    
    parser.add_argument('--dataset', type=str, default='data/word2vec.db', help='Path to the dataset to use for training')
    parser.add_argument('--vocab', type=str, default='data/vocab.txt', help='Path to the vocabulary file')
    
    
    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    
    output_file = args.output
    if output_file != 'stdout':
        logging.basicConfig(filename=output_file, filemode='w', level=log_level)
    else:
        logging.basicConfig(level=log_level)
    
    main(args)