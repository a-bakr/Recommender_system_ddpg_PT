"""
Actor network implementation for the DDPG algorithm.

This module implements the actor network used in the Deep Deterministic Policy Gradient (DDPG)
algorithm. The actor network is responsible for learning the policy that maps states to actions
(recommendations) in the recommender system.

The actor consists of:
1. A neural network that maps state representations to action vectors
2. A target network for stable learning
3. Methods for training and updating the networks
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ActorNetwork(nn.Module):
    """
    Neural network for the actor in DDPG.
    
    This network takes a state representation and outputs an action vector.
    The architecture consists of three fully connected layers with ReLU activations
    and a final Tanh activation to bound the output.
    
    Architecture:
        Input (3k) -> FC[ReLU] -> h
        h -> FC[ReLU] -> h
        h -> FC[Tanh] -> k
    
    where:
        k = embedding_dim
        h = hidden_dim
    """
    def __init__(self, embedding_dim, hidden_dim):
        """
        Initialize the actor network.
        
        Args:
            embedding_dim (int): Dimension of the embedding vectors
            hidden_dim (int): Dimension of the hidden layers
        """
        super(ActorNetwork, self).__init__()
        
        # Define the network architecture
        self.fc = nn.Sequential(
            nn.Linear(3 * embedding_dim, hidden_dim),  # First layer: 3k -> h
            nn.ReLU(),                                # ReLU activation
            nn.Linear(hidden_dim, hidden_dim),        # Second layer: h -> h
            nn.ReLU(),                                # ReLU activation
            nn.Linear(hidden_dim, embedding_dim),     # Third layer: h -> k
            nn.Tanh()                                 # Tanh activation for bounded output
        )    
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input state tensor of shape (batch_size, 3 * embedding_dim)
            
        Returns:
            torch.Tensor: Action vector of shape (batch_size, embedding_dim)
        """
        return self.fc(x)
    
    
class Actor:
    """
    Actor class that manages the actor network and its training.
    
    This class handles:
    1. Network initialization and target network setup
    2. Network updates and training
    3. Weight saving and loading
    """
    
    def __init__(self, embedding_dim, hidden_dim, learning_rate, tau):
        """
        Initialize the actor.
        
        Args:
            embedding_dim (int): Dimension of the embedding vectors
            hidden_dim (int): Dimension of the hidden layers
            learning_rate (float): Learning rate for the optimizer
            tau (float): Soft update parameter for target network
        """
        self.embedding_dim = embedding_dim
        
        # Initialize main and target networks
        self.network = ActorNetwork(embedding_dim, hidden_dim)
        self.target_network = ActorNetwork(embedding_dim, hidden_dim)
        self.target_network.load_state_dict(self.network.state_dict())  # Initialize target network
        
        # Set up optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Soft update parameter
        self.tau = tau
    
    def build_networks(self):
        """
        Build the networks by performing a dummy forward pass.
        This ensures all layers are properly initialized.
        """
        # Initialize networks with dummy input
        self.network(torch.zeros(1, 3 * self.embedding_dim))
        self.target_network(torch.zeros(1, 3 * self.embedding_dim))
    
    def update_target_network(self):
        """
        Perform soft update of target network parameters.
        
        Updates target network parameters using the formula:
        θ_target = τ * θ + (1 - τ) * θ_target
        where τ is the soft update parameter.
        """
        for target_param, param in zip(self.target_network.parameters(), self.network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
    def train(self, states, dq_das):
        """
        Train the actor network using the deterministic policy gradient.
        
        Args:
            states (torch.Tensor): Batch of state representations
            dq_das (torch.Tensor): Gradient of Q-value with respect to actions
        """
        self.optimizer.zero_grad()
        outputs = self.network(states)  # Get actions from current policy
        loss = -(outputs * dq_das).mean()  # Policy gradient loss
        loss.backward()  # Compute gradients
        self.optimizer.step()  # Update network parameters
        
    def save_weights(self, path: str):
        """
        Save the network weights to a file.
        
        Args:
            path (str): Path to save the weights
        """
        torch.save(self.network.state_dict(), path)
        
    def load_weights(self, path: str):
        """
        Load network weights from a file.
        
        Args:
            path (str): Path to load the weights from
        """
        self.network.load_state_dict(torch.load(path))
        

if __name__ == '__main__':
    # Example usage and testing
    embedding_dim = 128
    hidden_dim = 256
    
    learning_rate = 1e-3
    state_size = 3 * embedding_dim
    tau = 1e-3
    
    # Create and initialize actor
    actor = Actor(embedding_dim, hidden_dim, learning_rate, state_size, tau)
    actor.build_networks()
    
    # Test training with dummy data
    x = torch.rand(32, 3 * embedding_dim)  # Random states
    dq_das = torch.rand(32, embedding_dim)  # Random gradients
    
    actor.train(x, dq_das)
    
    
