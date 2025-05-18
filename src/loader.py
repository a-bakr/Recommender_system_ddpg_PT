import os
import numpy as np
import pandas as pd

def load_dataset(data_dir: str, mode: str):
    """
    Requires files items.csv, final_user_dict.npy, final_users_history_len.npy in DATA_DIR

    return:
    - users_num : Number of users to be used for train(eval)
    - total_items_num : Total number of items
    - users_dict : Dictionary in the format {user_id: [movie1, movie2, ...]}
    - users_history_lens : Length of history for each user (for all users)
    - movies_id_to_movies : Dictionary in the format {movie_id: movie_title} -> No data, not used
    """

    assert mode in ['train', 'eval'], "mode should be either 'train' or 'eval'"

    print('Interact Data loading...')

    movies_df = pd.read_csv(os.path.join(data_dir, 'items.csv'), dtype=int, header=None)
    movies_df.columns = ['MovieID']
    
    print("Data loading complete!")
    print("Data preprocessing...")

    # Map movie IDs to movie titles
    movies_id_to_movies = {movie[0]: movie[1:] for movie in movies_df.values}

    # Organize movies watched by each user in order
    users_dict = np.load(data_dir + 'final_user_dict.npy', allow_pickle=True)

    # Movie history length for each user
    users_history_lens = np.load(data_dir + 'final_users_history_len.npy')

    total_users_num = len(users_dict.item()) 
    total_items_num = len(movies_df)
    print(f"total users_num : {total_users_num}, total items_num : {total_items_num}")

    train_users_num = int(total_users_num * 0.8)

    if mode == 'train':
        users_num = train_users_num
        users_dict = {k: users_dict.item().get(k) for k in range(users_num)}
        # users_history_lens = users_history_lens[:users_num]
        print(f"train_users_num : {users_num}")
    
    elif mode == 'eval':
        users_num = total_users_num - train_users_num 
        users_dict = {k: users_dict.item().get(k) for k in range(train_users_num, total_users_num)}
        # users_history_lens = users_history_lens[-users_num:]
        print(f"eval_users_num : {users_num}")
    else :
        raise ValueError("Invalid mode")
    
    print("Done")

    return users_num, total_items_num, users_dict, users_history_lens, movies_id_to_movies 

def load_dataset_session(DATA_DIR: str, mode: str="train"):
    """
    Requires files items.csv, final_user_dict.npy, final_users_history_len.npy in DATA_DIR

    return:
    - users_num : Total number of users
    - total_items_num : Total number of items
    - users_dict : Dictionary in the format {user_id: [movie1, movie2, ...]}
    - users_history_lens : Length of history for each user (for all users)
    - movies_id_to_movies : Dictionary in the format {movie_id: movie_title} -> No data, not used
    """

    assert mode in ['train', 'eval'], "mode should be either 'train' or 'eval'"

    print('Interact Data loading...')

    movies_df = pd.read_csv(os.path.join(DATA_DIR, 'items.csv'), dtype=int, header=None)
    movies_df.columns = ['MovieID']
    
    print("Data loading complete!")
    print("Data preprocessing...")

    # Map movie IDs to movie titles
    movies_id_to_movies = {movie[0]: movie[1:] for movie in movies_df.values}

    # Organize movies watched by each user in order
    users_dict = np.load(DATA_DIR + 'final_user_dict.npy', allow_pickle=True)

    total_users_num = len(users_dict.item()) 
    total_items_num = len(movies_df)
    print(f"total users_num : {total_users_num}, total items_num : {total_items_num}")

    # Split user history: first 20% for eval, remaining 80% for training
    train_users_dict = {}
    eval_users_dict = {}

    for userid, movie_ratings in users_dict.item().items():
        split_index = int(len(movie_ratings) * 0.2)
        eval_users_dict[userid] = movie_ratings[:split_index]
        train_users_dict[userid] = movie_ratings[split_index:]

    if mode == 'train':
        users_dict = train_users_dict
        users_history_lens = np.load(DATA_DIR + 'final_train_users_history_len.npy')
    
    else:
        users_dict = eval_users_dict
        users_history_lens = np.load(DATA_DIR + 'final_eval_users_history_len.npy')

    return total_users_num, total_items_num, users_dict, users_history_lens, movies_id_to_movies  

if __name__ == "__main__":
    ROOT_DIR = os.getcwd()
    data_dir = os.path.join(ROOT_DIR, 'data/ml-1m/')
    eval_users_num, _, test_users_dict, test_users_history_lens, _ = load_dataset(data_dir, 'test')