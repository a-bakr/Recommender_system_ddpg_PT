import numpy as np
import random
from src.tree import SumTree, MinTree
import torch

class PriorityExperienceReplay(object):
    """
    Priority Experience Replay (PER) buffer for reinforcement learning.
    
    This implementation uses a SumTree and MinTree to efficiently sample experiences
    based on their priorities. Experiences with higher TD errors are sampled more frequently.
    
    Attributes:
        buffer_size (int): Maximum number of experiences to store
        embedding_dim (int): Dimension of the embedding vectors
        states (np.ndarray): Buffer for state vectors
        actions (np.ndarray): Buffer for action vectors  
        rewards (np.ndarray): Buffer for reward values
        next_states (np.ndarray): Buffer for next state vectors
        dones (np.ndarray): Buffer for done flags
        sum_tree (SumTree): Tree structure for priority sampling
        min_tree (MinTree): Tree structure for tracking minimum priority
        max_priority (float): Maximum priority value seen so far
        alpha (float): Priority exponent parameter
        beta (float): Importance sampling exponent parameter
        beta_constant (float): Rate at which beta increases
    """

    def __init__(self, buffer_size: int, embedding_dim: int):
        """
        Initialize the priority replay buffer.
        
        Args:
            buffer_size: Maximum number of experiences to store
            embedding_dim: Dimension of the embedding vectors
        """
        self.buffer_size = buffer_size
        self.crt_idx = 0
        self.is_full = False
        
        # Initialize experience buffers
        self.states = np.zeros((buffer_size, 3 * embedding_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, embedding_dim), dtype=np.float32)
        self.rewards = np.zeros((buffer_size), dtype=np.float32)
        self.next_states = np.zeros((buffer_size, 3 * embedding_dim), dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.bool)

        # Initialize priority trees
        self.sum_tree = SumTree(buffer_size)
        self.min_tree = MinTree(buffer_size)

        # Initialize hyperparameters
        self.max_priority = 1.0
        self.alpha = 0.6  # Priority exponent
        self.beta = 0.4   # Importance sampling exponent
        self.beta_constant = 0.00001  # Beta increase rate

    def append(self, state: np.ndarray, action: np.ndarray, reward: float, 
              next_state: np.ndarray, done: bool) -> None:
        """
        Add a new experience to the buffer.
        
        Args:
            state: Current state vector
            action: Action vector taken
            reward: Reward received
            next_state: Next state vector
            done: Whether episode ended
        """
        self.states[self.crt_idx] = state
        self.actions[self.crt_idx] = action
        self.rewards[self.crt_idx] = reward
        self.next_states[self.crt_idx] = next_state
        self.dones[self.crt_idx] = done

        # Add experience to priority trees with current max priority
        priority = self.max_priority ** self.alpha
        self.sum_tree.add_data(priority)
        self.min_tree.add_data(priority)
        
        self.crt_idx = (self.crt_idx + 1) % self.buffer_size
        if self.crt_idx == 0:
            self.is_full = True

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, 
                                              torch.Tensor, np.ndarray, np.ndarray, list]:
        """
        Sample a batch of experiences based on their priorities.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple containing:
            - Batch of states as torch tensor
            - Batch of actions as torch tensor  
            - Batch of rewards as torch tensor
            - Batch of next states as torch tensor
            - Batch of done flags as numpy array
            - Importance sampling weights as numpy array
            - Indices of sampled experiences
        """
        rd_idx = []
        weight_batch = []
        index_batch = []
        sum_priority = self.sum_tree.sum_all_priority()
        
        N = self.buffer_size if self.is_full else self.crt_idx
        min_priority = self.min_tree.min_priority() / sum_priority
        max_weight = (N * min_priority) ** (-self.beta)

        segment_size = sum_priority / batch_size
        
        for j in range(batch_size):
            min_seg = segment_size * j
            max_seg = segment_size * (j + 1)

            random_num = random.uniform(min_seg, max_seg)
            priority, tree_index, buffer_index = self.sum_tree.search(random_num)
            rd_idx.append(buffer_index)

            p_j = priority / sum_priority
            w_j = (p_j * N) ** (-self.beta) / max_weight
            weight_batch.append(w_j)
            index_batch.append(tree_index)
        
        self.beta = min(1.0, self.beta + self.beta_constant)

        # Sample experiences from buffers
        batch_states = self.states[rd_idx]
        batch_actions = self.actions[rd_idx]
        batch_rewards = self.rewards[rd_idx]
        batch_next_states = self.next_states[rd_idx]
        batch_dones = self.dones[rd_idx]

        return (torch.from_numpy(batch_states), torch.from_numpy(batch_actions),
                torch.from_numpy(batch_rewards), torch.from_numpy(batch_next_states),
                batch_dones, np.array(weight_batch), index_batch)

    def update_priority(self, priority: float, index: int) -> None:
        """
        Update the priority of an experience.
        
        Args:
            priority: New priority value
            index: Index of experience to update
        """
        priority_alpha = priority ** self.alpha
        self.sum_tree.update_priority(priority_alpha, index)
        self.min_tree.update_priority(priority_alpha, index)
        self.update_max_priority(priority_alpha)

    def update_max_priority(self, priority: float) -> None:
        """
        Update the maximum priority seen so far.
        
        Args:
            priority: New priority value to compare against
        """
        self.max_priority = max(self.max_priority, priority)
