"""
Critic network implementation for the DDPG algorithm.

This module implements the critic network used in the Deep Deterministic Policy Gradient (DDPG)
algorithm. The critic network is responsible for learning the Q-value function that estimates
the expected return for state-action pairs.

The critic consists of:
1. A neural network that maps state-action pairs to Q-values
2. A target network for stable learning
3. Methods for training and updating the networks
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class CriticNetwork(nn.Module):
    """
    Neural network for the critic in DDPG.
    
    This network takes a state-action pair and outputs a Q-value estimate.
    The architecture processes the state and action separately before combining them.
    
    Architecture:
        State (3k) -> FC1[ReLU] -> k
        Action (k) + State (k) -> CONCAT -> 2k
        2k -> FC2[ReLU] -> h
        h -> FC3[ReLU] -> h
        h -> HEAD -> 1
    
    where:
        k = embedding_dim
        h = hidden_dim
    """
    def __init__(self, embedding_dim, hidden_dim):
        """
        Initialize the critic network.
        
        Args:
            embedding_dim (int): Dimension of the embedding vectors
            hidden_dim (int): Dimension of the hidden layers
        """
        super(CriticNetwork, self).__init__()
        
        # Define the network architecture
        self.fc1 = nn.Linear(3 * embedding_dim, embedding_dim)  # Process state
        self.fc2 = nn.Linear(2 * embedding_dim, hidden_dim)     # Process combined state-action
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)           # Deep representation
        self.head = nn.Linear(hidden_dim, 1)                   # Output Q-value
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (tuple): Tuple of (action, state) tensors
                - action: shape (batch_size, embedding_dim)
                - state: shape (batch_size, 3 * embedding_dim)
            
        Returns:
            torch.Tensor: Q-value estimate of shape (batch_size, 1)
        """
        a, s = x[0], x[1]  # Unpack action and state
        
        # Process state through first layer
        s = torch.relu(self.fc1(s))
        # Concatenate processed state with action
        s = torch.cat([a, s], dim=1)
        # Process through remaining layers
        s = torch.relu(self.fc2(s))
        s = torch.relu(self.fc3(s))
        Q = self.head(s)
        return Q  # Q-value used for item selection

class Critic:
    """
    Critic class that manages the critic network and its training.
    
    This class handles:
    1. Network initialization and target network setup
    2. Q-value computation and gradient calculation
    3. Network updates and training
    4. Weight saving and loading
    """
    
    def __init__(self, hidden_dim, learning_rate, embedding_dim, tau):
        """
        Initialize the critic.
        
        Args:
            hidden_dim (int): Dimension of the hidden layers
            learning_rate (float): Learning rate for the optimizer
            embedding_dim (int): Dimension of the embedding vectors
            tau (float): Soft update parameter for target network
        """
        self.embedding_dim = embedding_dim

        # Initialize main and target networks
        self.network = CriticNetwork(embedding_dim, hidden_dim)
        self.target_network = CriticNetwork(embedding_dim, hidden_dim)
        self.target_network.load_state_dict(self.network.state_dict())
        self.target_network.eval()  # Set target network to evaluation mode
        
        # Set up optimizer and loss function
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss(reduction='none')  # Use no reduction for weighted loss

        # Soft update parameter
        self.tau = tau

    def build_networks(self):
        """
        Build the networks by performing a dummy forward pass.
        This ensures all layers are properly initialized.
        """
        # Initialize networks with dummy input
        self.network([torch.zeros(1, self.embedding_dim), torch.zeros(1, 3 * self.embedding_dim)])
        self.target_network([torch.zeros(1, self.embedding_dim), torch.zeros(1, 3 * self.embedding_dim)])
    
    def update_target_network(self):
        """
        Perform soft update of target network parameters.
        
        Updates target network parameters using the formula:
        θ_target = τ * θ + (1 - τ) * θ_target
        where τ is the soft update parameter.
        """
        for target_param, param in zip(self.target_network.parameters(), self.network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
    def dq_da(self, inputs):
        """
        Compute the gradient of Q-value with respect to actions.
        
        Args:
            inputs (tuple): Tuple of (action, state) tensors
            
        Returns:
            torch.Tensor: Gradient of Q-value with respect to actions
        """
        actions = inputs[0]
        states = inputs[1]
        actions = torch.tensor(actions, requires_grad=True)
        outputs = self.network([actions, states])
        outputs.backward(torch.ones_like(outputs))
        return actions.grad

    def train(self, inputs, td_targets, weight_batch):
        """
        Train the critic network using TD learning.
        
        Args:
            inputs (tuple): Tuple of (action, state) tensors
            td_targets (torch.Tensor): Target Q-values
            weight_batch (torch.Tensor): Weights for each sample in the batch
            
        Returns:
            float: Weighted loss value
        """
        weight_batch = torch.tensor(weight_batch, dtype=torch.float32)
        self.optimizer.zero_grad()
        outputs = self.network(inputs)
        
        # Compute weighted loss
        loss = self.loss_fn(outputs, td_targets)
        weighted_loss = (loss * weight_batch).mean()
        weighted_loss.backward()
        self.optimizer.step()
        return weighted_loss.item()

    def train_on_batch(self, inputs, td_targets, weight_batch):
        """
        Alternative training method that combines zero_grad, forward pass,
        loss computation, and optimization in one step.
        
        Args:
            inputs (tuple): Tuple of (action, state) tensors
            td_targets (torch.Tensor): Target Q-values
            weight_batch (torch.Tensor): Weights for each sample in the batch
            
        Returns:
            float: Weighted loss value
        """
        self.optimizer.zero_grad()
        outputs = self.network(inputs)
        loss = self.loss_fn(outputs, td_targets)
        weighted_loss = (loss * weight_batch).mean()
        weighted_loss.backward()
        self.optimizer.step()
        return weighted_loss.item()
            
    def save_weights(self, path):
        """
        Save the network weights to a file.
        
        Args:
            path (str): Path to save the weights
        """
        torch.save(self.network.state_dict(), path)
        
    def load_weights(self, path):
        """
        Load network weights from a file.
        
        Args:
            path (str): Path to load the weights from
        """
        self.network.load_state_dict(torch.load(path))