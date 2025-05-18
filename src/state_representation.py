import torch
import torch.nn as nn


class DRRAveStateRepresentation(nn.Module):
    """
    Deep Reinforcement Recommender with Average State Representation.
    
    This module implements a state representation method for a deep reinforcement learning
    based recommender system. It combines user embeddings with weighted item embeddings
    to create a rich state representation.
    
    Attributes:
        embedding_dim (int): Dimension of the embedding vectors
        wav (nn.Conv1d): 1D convolution layer for processing item embeddings
        flatten (nn.Flatten): Layer to flatten tensors
        weights (torch.Tensor): Predefined weights for item importance
    """
    
    def __init__(self, embedding_dim, state_size=10):
        """
        Initialize the DRR state representation module.
        
        Args:
            embedding_dim (int): Dimension of the embedding vectors
            state_size (int, optional): Number of items in state. Defaults to 10.
        """
        super(DRRAveStateRepresentation, self).__init__()
        self.embedding_dim = embedding_dim        
        self.wav = nn.Conv1d(in_channels=state_size, out_channels=1, kernel_size=1)
        self.flatten = nn.Flatten()
        
        # Predefined weights for item importance (sums to 1.0)
        self.weights = torch.tensor([0.19, 0.17, 0.15, 0.13, 0.11, 0.09, 0.07, 0.05, 0.03, 0.01])
        
    def forward(self, x):
        """
        Forward pass of the state representation.
        
        Args:
            x (list): List containing:
                - x[0]: User embeddings tensor of shape (batch_size, embedding_dim)
                - x[1]: Item embeddings tensor of shape (batch_size, state_size, embedding_dim)
                
        Returns:
            torch.Tensor: Concatenated state representation of shape 
                         (batch_size, 3 * embedding_dim)
        """
        items_eb = x[1] 
        
        # Apply importance weights to each item embedding
        for i in range(10):
            items_eb[:,i,:] = items_eb[:,i,:] * self.weights[i]
        
        # Calculate weighted average of item embeddings
        avp = torch.sum(items_eb, dim=1).squeeze(1)
        
        # Combine user embeddings with weighted item embeddings
        user_avp = x[0] * avp
        
        # Concatenate user embeddings, user-item interaction, and item embeddings
        concat = torch.cat([x[0], user_avp, avp], dim=1)
        return concat


if __name__ == '__main__':
    embedding_dim = 128
    model = DRRAveStateRepresentation(embedding_dim)
    x0 = torch.rand(32, embedding_dim)  # Example user embeddings
    x1 = torch.rand(32, 10, embedding_dim)  # Example item embeddings
    output = model([x0, x1])
    print(output.shape)
    for name, param in model.named_parameters():
        print(name)