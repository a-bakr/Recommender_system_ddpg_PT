"""
Offline environment for training and evaluating the recommender system.

This module implements an offline environment that simulates user-movie interactions
for training and evaluating the recommender system. It maintains state using the
user's most recently watched movies and provides rewards based on whether the
recommended movies match the user's preferences.

The environment follows these key principles:
    1. State representation using recent movie history
    2. Reward calculation based on user ratings
    3. Episode termination based on recommendation limits
    4. Support for both single and top-k recommendations
"""

import numpy as np


class OfflineEnv(object):
    """
    Offline environment for training and evaluating the recommender system.
    
    This environment simulates user-movie interactions in an offline setting, where
    the agent recommends movies to users based on their viewing history. The environment
    maintains state using the user's most recently watched movies and provides rewards
    based on whether the recommended movies match the user's preferences.
    
    Attributes:
        users_dict (dict): Maps user IDs to their movie watching history and ratings
        users_history_lens (int): Length of viewing history for each user
        items_id_to_name (dict): Maps movie IDs to their titles and genres
        state_size (int): Number of recent movies to use for state representation
        available_users (list): List of users with sufficient viewing history
        user (int): Current user being served recommendations
        user_items (dict): Current user's movie ratings
        items (list): Current state - list of recent movie IDs
        done (bool): Whether the episode is complete
        recommended_items (set): Set of movies already recommended
        done_count (int): Maximum trajectory length
    """
    
    def __init__(self, 
                users_dict: dict[int, list[tuple[int, int]]],
                users_history_lens: int,
                movies_id_to_movies: dict[str, tuple[str, str]],
                state_size: int,
                fix_user_id: int = None):
        """
        Initialize the offline environment.
        
        Args:
            users_dict: Maps user IDs to lists of (movie_id, rating) tuples
            users_history_lens: Length of viewing history for each user
            movies_id_to_movies: Maps movie IDs to (title, genre) tuples
            state_size: Number of recent movies to use for state
            fix_user_id: Optional fixed user ID for evaluation
        """
        # Store user data and movie information
        self.users_dict = users_dict                    # User viewing history and ratings
        self.users_history_lens = users_history_lens    # Length of each user's history
        self.items_id_to_name = movies_id_to_movies     # Movie metadata
        
        # Initialize environment parameters
        self.state_size = state_size                    # Number of movies in state
        self.available_users = self._generate_available_users()  # Users with sufficient history

        # Set up for evaluation if user ID is fixed
        self.fix_user_id = fix_user_id

        # Initialize episode state
        self.user = fix_user_id if fix_user_id else np.random.choice(self.available_users)
        self.user_items = {data[0]:data[1] for data in self.users_dict[self.user]}  # Current user's ratings
        self.items = [data[0] for data in self.users_dict[self.user][:self.state_size]]  # Initial state
        self.done = False
        self.recommended_items = set(self.items)  # Track recommended movies
        self.done_count = 3000  # Maximum recommendations per episode
        
    def _generate_available_users(self) -> list[int]:
        """
        Generate list of users with sufficient viewing history.
        
        Returns:
            List of user IDs with history length greater than state_size
        """
        available_users = []
        for i, length in zip(self.users_dict.keys(), self.users_history_lens):
            if length > self.state_size:
                available_users.append(i)
        return available_users
    
    def reset(self) -> tuple[int, list[int], bool]:
        """
        Reset the environment for a new episode.
        
        This method:
        1. Selects a new user (random or fixed)
        2. Initializes the user's movie ratings
        3. Sets up the initial state with recent movies
        4. Resets tracking variables
        
        Returns:
            Tuple of (user_id, initial_state, done)
        """
        # Select user and initialize their data
        self.user = self.fix_user_id if self.fix_user_id else np.random.choice(self.available_users)
        self.user_items = {data[0]: data[1] for data in self.users_dict[self.user]}
        self.items = [data[0] for data in self.users_dict[self.user][:self.state_size]]
        
        # Reset episode tracking
        self.done = False
        self.recommended_items = set(self.items)
        return self.user, self.items, self.done
        
    def step(self, action: int, top_k: bool = False) -> tuple[list[int], float, bool, set[int]]:
        """
        Take a step in the environment by recommending a movie.
        
        This method:
        1. Processes the recommendation(s)
        2. Calculates rewards based on user ratings
        3. Updates the state with new movies
        4. Checks termination conditions
        
        Args:
            action: Movie ID to recommend (or list of IDs for top-k)
            top_k: Whether to use top-k recommendation mode
            
        Returns:
            Tuple of (new_state, reward, done, recommended_items)
        """
        # Default negative reward for incorrect recommendations
        reward = -0.5

        if top_k:
            # Lists to track successful recommendations and their rewards
            correctly_recommended = []
            rewards = []
            
            # Process each action in the top-k list
            for act in action:
                # Check if movie exists in user's history and hasn't been recommended
                if act in self.user_items.keys() and act not in self.recommended_items:
                    correctly_recommended.append(act)
                    # Calculate reward based on user's rating (normalized to [-1,1])
                    rewards.append((self.user_items[act] - 3)/2)
                else:
                    rewards.append(-0.5)
                # Track all recommended items
                self.recommended_items.add(act)
            
            # Update state if at least one recommendation was successful
            if max(rewards) > 0:
                self.items = self.items[len(correctly_recommended):] + correctly_recommended
            reward = rewards

        else:
            # Single recommendation mode
            # Check if movie exists in user's history and hasn't been recommended
            if action in self.user_items.keys() and action not in self.recommended_items:
                # Calculate reward based on user's rating (normalized to [-1,1])
                reward = self.user_items[action] - 3
            
            # Update state if recommendation was successful
            if reward > 0:
                # Remove the first item from the state and append the new recommended movie
                # This maintains a fixed-size state window by sliding it forward
                self.items = self.items[1:] + [action]
            # Track recommended item
            self.recommended_items.add(action)

        # Check termination conditions:
        # 1. Exceeded maximum number of recommendations
        # 2. Recommended all movies in user's history
        if len(self.recommended_items) > self.done_count or len(self.recommended_items) >= self.users_history_lens[self.user-1]:
            self.done = True
            
        return self.items, reward, self.done, self.recommended_items

    def get_items_names(self, items_ids: list[int]) -> list[list[str]]:
        """
        Get movie titles and genres for given movie IDs.
        
        This method looks up the metadata for each movie ID and returns
        a list of [title, genre] pairs. If a movie ID is not found,
        it returns ['Not in list'] for that movie.
        
        Args:
            items_ids: List of movie IDs to look up
            
        Returns:
            List of [title, genre] pairs for each movie ID
        """
        items_names = []
        for id in items_ids:
            try:
                items_names.append(self.items_id_to_name[str(id)])
            except:
                items_names.append(list(['Not in list']))
        return items_names