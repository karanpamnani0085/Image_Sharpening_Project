import torch
import random
import numpy as np
import os

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_checkpoint(state, filename):
    """Save checkpoint"""
    torch.save(state, filename)
    print(f"Saved checkpoint: {filename}")

def load_checkpoint(filename, model, optimizer=None):
    """Load checkpoint"""
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['epoch']

def get_device():
    """Get device for training"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')