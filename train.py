"""
Training script for the Deep Reinforcement Learning based Recommender System.

This script implements the training loop for the DDPG-based recommender system.
It handles environment setup, agent initialization, and the training process.

The training process follows these steps:
1. Load and preprocess the dataset
2. Initialize the offline environment
3. Create the DDPG agent
4. Train the agent through episodes
"""

# Dependencies
import pandas as pd
import numpy as np
import torch
import itertools
import matplotlib.pyplot as plt
import time

from src.envs import OfflineEnv
from src.recommender import DRRAgent
from src.loader import load_dataset, load_dataset_session
import os

"""
[Training Method]
- Randomly select a user at each episode (OfflineEnv.user)
- Create state using the 10 most recently watched movies by the user
- Recommend movies excluding the 10 most recently watched movies (DRRAgent.recommend_item())
- Maximum trajectory length per user is about 3000, ending when recommendations reach the length of user's movie history (OfflineEnv.step())
- Actor and Critic parameter updates are done in batches of 32 from the replay buffer
"""

def trainer(args):
    """
    Main training function for the recommender system.
    
    This function handles the complete training pipeline:
    1. Loads and preprocesses the dataset
    2. Initializes the offline environment
    3. Creates the DDPG agent
    4. Trains the agent through episodes
    
    Args:
        args: Command line arguments containing training parameters:
            - state_size: Size of the state representation
            - mode: Training mode ('train' or 'eval')
            - use_wandb: Whether to use Weights & Biases for logging
            - dim_emb: Embedding dimension
            - dim_actor: Actor network hidden dimension
            - lr_actor: Actor learning rate
            - dim_critic: Critic network hidden dimension
            - lr_critic: Critic learning rate
            - discount: Discount factor for future rewards
            - tau: Target network update rate
            - memory_size: Size of the replay buffer
            - batch_size: Batch size for training
            - epsilon: Exploration rate
            - std: Standard deviation for exploration noise
            - max_episode_num: Maximum number of training episodes
            - checkpoint: Path to checkpoint for loading pretrained model
            - top_k: Number of top items to recommend
    """
    ROOT_DIR = os.getcwd()
    DATA_DIR = os.path.join(ROOT_DIR, 'data/ml-1m/')
    
    # Load dataset and create user history
    total_users_num, total_items_num, train_users_dict, users_history_lens, movies_id_to_movies = load_dataset_session(DATA_DIR, 'train')
    
    time.sleep(2)
    
    # Initialize offline environment
    env = OfflineEnv(train_users_dict,
                    users_history_lens,
                    movies_id_to_movies,
                    args.state_size)
    
    # Initialize DDPG agent
    recommender = DRRAgent(env = env,
                            users_num = total_users_num,
                            items_num = total_items_num,
                            state_size = args.state_size,
                            is_eval = args.mode == 'eval',
                            use_wandb = args.use_wandb,
                            embedding_dim = args.dim_emb,
                            actor_hidden_dim = args.dim_actor,
                            actor_learning_rate = args.lr_actor,
                            critic_hidden_dim = args.dim_critic,
                            critic_learning_rate = args.lr_critic,
                            discount = args.discount,
                            tau = args.tau,
                            memory_size = args.memory_size,
                            batch_size = args.batch_size,
                            epsilon = args.epsilon,
                            std = args.std,
                            args = args,
                           )
    
    # Start training
    recommender.train(max_episode_num = args.max_episode_num, 
                      load_model = args.checkpoint, 
                      top_k=args.top_k)

# if __name__ == '__main__':
#     trainer()