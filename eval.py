"""
Evaluation script for the Deep Reinforcement Learning based Recommender System.

This script implements the offline evaluation of the trained recommender system.
It evaluates the model's performance using precision@k and NDCG@k metrics.

The evaluation process follows these steps:
1. Load test dataset and create evaluation environment
2. For each user in the test set:
   - Generate recommendations using the trained policy
   - Calculate precision and NDCG metrics
3. Report average performance across all evaluated users
"""

import numpy as np
import time
import os
import torch
import torch.nn.functional as F
import pandas as pd

from src.envs import OfflineEnv
from src.recommender import DRRAgent
from src.loader import load_dataset, load_dataset_session
import tensorflow as tf

"""
[Evaluation Method - Offline Evaluation (Algorithm 2)]
- Evaluate one by one from eval_user_list
- At each time step, take action with the trained policy and recommend items -> observe reward, update state, and remove recommended items from recommendation list
- Need to decide how many recommendations to make per user (In the Jupyter notebook it seems to do it once, but according to the algorithm it averages over T times)
"""

ROOT_DIR = os.getcwd()
DATA_DIR = os.path.join(ROOT_DIR, 'data/ml-1m/')

def evaluate(recommender, env, args, check_movies: bool=False, top_k: int=1, length: int=1):
    """
    Evaluate the recommender system for a single user.
    
    This function runs the evaluation loop for one user, making recommendations
    and calculating precision and NDCG metrics.
    
    Args:
        recommender: Trained DRRAgent instance
        env: OfflineEnv instance for evaluation
        args: Command line arguments
        check_movies (bool): Whether to print detailed movie information
        top_k (int): Number of items to recommend at each step
        length (int): Maximum number of recommendation steps
        
    Returns:
        tuple: (mean_precision, mean_ndcg, mean_reward)
            - mean_precision: Average precision@k
            - mean_ndcg: Average NDCG@k
            - mean_reward: Average reward per step
    """
    mean_precision = 0
    mean_ndcg = 0

    episode_reward = 0
    steps = 0

    user_id, items_ids, done = env.reset()
    print(f"[STARTING RECOMMENDATION TO USER {user_id}]")
    if check_movies:
        print(f'user_id : {user_id}, rated_items_length:{len(env.user_items)}')

    while not done:
        
        if not args.modality:
            user_eb = recommender.embedding_network.u_embedding(torch.tensor([user_id], dtype=torch.long))
            items_eb = recommender.embedding_network.m_embedding(torch.tensor(items_ids, dtype=torch.long))
        else :
            tf_items_ids = tf.convert_to_tensor(items_ids, dtype=tf.int32)
            tf_user_id = tf.convert_to_tensor(user_id, dtype=tf.int32)
            user_eb, items_eb = recommender.embedding_network.get_embedding([tf_user_id, tf_items_ids])
            user_eb, items_eb = tf.reshape(user_eb, (1,args.dim_emb)).numpy(), items_eb.numpy()
            
        state = recommender.srm_ave([
            torch.tensor(user_eb, dtype=torch.float32),
            torch.tensor(items_eb, dtype=torch.float32).unsqueeze(0)
            ])

        action = recommender.actor.network(state)

        recommended_item = recommender.recommend_item(action, env.recommended_items, top_k=top_k)

        next_items_ids, reward, done, _ = env.step(recommended_item, top_k=top_k)

        if check_movies:
            print(f'\t[step: {steps+1}] recommended items ids : {recommended_item}, reward : {reward}')

        correct_list = [1 if r > 0 else 0 for r in reward]

        dcg, idcg = calculate_ndcg(correct_list, [1 for _ in range(len(reward))])        
        mean_ndcg += dcg/idcg

        correct_num = len(reward) - correct_list.count(0)
        mean_precision += correct_num / len(reward)

        reward = np.sum(reward)
        items_ids = next_items_ids
        episode_reward += reward
        steps += 1

        if done or steps >= length:
            break

    if check_movies:
        print(f"\tprecision@{top_k} : {mean_precision/steps}, ndcg@{top_k} : {mean_ndcg/steps}, episode_reward : {episode_reward/steps}\n")

    return mean_precision/steps, mean_ndcg/steps, episode_reward/steps

def calculate_ndcg(rel, irel):
    """
    Calculate NDCG (Normalized Discounted Cumulative Gain) metric.
    
    Args:
        rel (list): List of relevance scores for recommended items
        irel (list): List of ideal relevance scores
        
    Returns:
        tuple: (dcg, idcg)
            - dcg: Discounted Cumulative Gain
            - idcg: Ideal Discounted Cumulative Gain
    """
    dcg = 0
    idcg = 0
    rel = [1 if r > 0 else 0 for r in rel]
    for i, (r, ir) in enumerate(zip(rel, irel)):
        dcg += (r)/np.log2(i+2)
        idcg += (ir)/np.log2(i+2)
    return dcg, idcg

def evaluater(args):
    """
    Main evaluation function for the recommender system.
    
    This function handles the complete evaluation pipeline:
    1. Loads test dataset
    2. Creates evaluation environment
    3. Evaluates the model on test users
    4. Reports average performance metrics
    
    Args:
        args: Command line arguments containing evaluation parameters:
            - state_size: Size of the state representation
            - mode: Evaluation mode
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
            - saved_actor: Path to saved actor model
            - saved_critic: Path to saved critic model
            - top_k: Number of top items to recommend
    """
    total_users_num, total_items_num, eval_users_dict, users_history_lens, movies_id_to_movies = load_dataset_session(DATA_DIR, 'eval')

    sum_precision, sum_ndcg = 0, 0

    end_evaluation = 50

    temp_env = OfflineEnv(eval_users_dict, users_history_lens, movies_id_to_movies, args.state_size)
    avaiable_users = temp_env.available_users
    print(f"Available number of users: {len(avaiable_users)}")

    for i, user_id in enumerate(avaiable_users):
        env = OfflineEnv(eval_users_dict, 
                            users_history_lens, 
                            movies_id_to_movies, 
                            args.state_size, 
                            fix_user_id=user_id)
        
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
        
        recommender.eval()
        recommender.load_model(args.saved_actor, args.saved_critic)
        
        precision, ndcg, _ = evaluate(
            recommender,
            env,
            args,
            check_movies=True,
            top_k=args.top_k,
            length=args.state_size)
        
        sum_precision += precision
        sum_ndcg += ndcg

        if i > end_evaluation:
            break

    print("\n[FINAL RESULT]")
    print(f'precision@{args.top_k} : {sum_precision/(end_evaluation)}, ndcg@{args.top_k} : {sum_ndcg/(end_evaluation)}')