from enum import Enum
from torch import nn
from src.data import word2vec
import torch
import numpy as np

class SimilarityClass(Enum):
    RAW = "raw",
    SQUARED = "squared",
    SQRT = "sqrt",
    DISCRETE = "discrete"

cos_sim = nn.CosineSimilarity(dim=1)

def similarity_function(target_list, guess_list):
    # Extract vectors for the first word and the guess list
    x = torch.tensor(np.array([word2vec[g] for g in target_list]))
    y = torch.tensor(np.array([word2vec[g] for g in guess_list]))
    
    similarities = cos_sim(x,y)
    
    return similarities

def similarity_to_reward(similarities, args):
    # Squaring curve mapping from [-1, 1] to [-1, 1]
    if args.similarity == SimilarityClass.SQRT:
        similarities = (similarities + 1) / 2
        similarities = similarities ** 0.5
        similarities = 2 * similarities - 1
        return similarities
    elif args.similarity == SimilarityClass.DISCRETE:
        # Map 1D tensor. Logic is if guess is correct, reward = 1. Else, reward = -1
        o = torch.tensor([1 if s > 0.99 else -0.2 for s in similarities])
        return o
    else:
        raise "Similarity class not implemented"