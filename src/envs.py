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
        self.users_dict = users_dict                    
        self.users_history_lens = users_history_lens    
        self.items_id_to_name = movies_id_to_movies     
        
        self.state_size = state_size                    
        self.available_users = self._generate_available_users()

        self.fix_user_id = fix_user_id

        self.user = fix_user_id if fix_user_id else np.random.choice(self.available_users)
        self.user_items = {data[0]:data[1] for data in self.users_dict[self.user]}
        self.items = [data[0] for data in self.users_dict[self.user][:self.state_size]]
        self.done = False
        self.recommended_items = set(self.items)
        self.done_count = 3000
        
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
        
        Returns:
            Tuple of (user_id, initial_state, done)
        """
        self.user = self.fix_user_id if self.fix_user_id else np.random.choice(self.available_users)
        self.user_items = {data[0]:data[1] for data in self.users_dict[self.user]}
        self.items = [data[0] for data in self.users_dict[self.user][:self.state_size]]
        self.done = False
        self.recommended_items = set(self.items)
        return self.user, self.items, self.done
        
    def step(self, action: int, top_k: bool = False) -> tuple[list[int], float, bool, set[int]]:
        """
        Take a step in the environment by recommending a movie.
        
        Args:
            action: Movie ID to recommend
            top_k: Whether to use top-k recommendation mode
            
        Returns:
            Tuple of (new_state, reward, done, recommended_items)
        """
        reward = -0.5

        if top_k:
            correctly_recommended = []
            rewards = []
            for act in action:
                if act in self.user_items.keys() and act not in self.recommended_items:
                    correctly_recommended.append(act)
                    rewards.append((self.user_items[act] - 3)/2)
                else:
                    rewards.append(-0.5)
                self.recommended_items.add(act)
            
            if max(rewards) > 0:
                self.items = self.items[len(correctly_recommended):] + correctly_recommended
            reward = rewards

        else:
            if action in self.user_items.keys() and action not in self.recommended_items:
                reward = self.user_items[action] - 3
            
            if reward > 0:
                self.items = self.items[1:] + [action]
            
            self.recommended_items.add(action)

        if len(self.recommended_items) > self.done_count or len(self.recommended_items) >= self.users_history_lens[self.user-1]:
            self.done = True
            
        return self.items, reward, self.done, self.recommended_items

    def get_items_names(self, items_ids: list[int]) -> list[list[str]]:
        """
        Get movie titles and genres for given movie IDs.
        
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