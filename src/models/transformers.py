import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer

class SemantleSolver(nn.Module):
    def __init__(self, hidden_dim, num_layers, pretrained_model='bert-base-uncased'):
        super(SemantleSolver, self).__init__()
        self.transformer = BertModel.from_pretrained(pretrained_model)
        self.context_layer = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.hidden_dim = hidden_dim
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)
        
    def forward(self, guess_word, context_vector):
        # Tokenize and get embeddings from BERT
        inputs = self.tokenizer(guess_word, return_tensors='pt')
        outputs = self.transformer(**inputs)
        guess_embedding = outputs.last_hidden_state.mean(dim=1)
        
        # Update context vector
        updated_context = self.context_layer(context_vector) + guess_embedding
        
        # Predict similarity score
        similarity_score = self.output_layer(updated_context)
        
        return similarity_score, updated_context

# Initialize model
hidden_dim = 768
num_layers = 12
model = SemantleSolver(hidden_dim, num_layers)

# Example usage
guess_word = "apple"
context_vector = torch.zeros((1, hidden_dim))  # Initial context vector

# Predict similarity score
similarity_score, updated_context = model(guess_word, context_vector)
print(similarity_score)