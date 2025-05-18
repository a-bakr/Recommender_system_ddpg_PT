import numpy as np


class SumTree:
    """
    A SumTree maintains a binary tree where each node represents the sum of its children.
    This is useful for sampling data points based on their priority, with higher priorities being more likely to be sampled.
    
    Attributes:
        buffer_size (int): Maximum number of data points that can be stored
        tree (np.ndarray): Array representing the binary tree structure
        index (int): Current position in the tree for adding new data
    """
    
    def __init__(self, buffer_size: int):
        """
        Initialize the SumTree.
        
        Args:
            buffer_size (int): Maximum number of data points that can be stored
        """
        self.buffer_size = buffer_size
        self.tree = np.zeros((buffer_size * 2 - 1))
        self.index = buffer_size - 1

    def __update_tree(self, index: int) -> None:
        """
        Update the tree values by propagating sums up from the given index.
        
        Args:
            index (int): Index to start updating from
        """
        while True:
            index = (index - 1) // 2
            left = (index * 2) + 1
            right = (index * 2) + 2
            self.tree[index] = self.tree[left] + self.tree[right]
            if index == 0:
                break

    def add_data(self, priority: float) -> None:
        """
        Add a new data point with the given priority.
        
        Args:
            priority (float): Priority value for the new data point
        """
        if self.index == self.buffer_size * 2 - 1:
            self.index = self.buffer_size - 1

        self.tree[self.index] = priority
        self.__update_tree(self.index)
        self.index += 1

    def search(self, num: float) -> tuple[float, int, int]:
        """
        Search for a data point based on a random number.
        
        Args:
            num (float): Random number to search with
            
        Returns:
            tuple containing:
                - Priority value at the found position
                - Tree index of the found position
                - Buffer index of the found position
        """
        current = 0
        while True:
            left = (current * 2) + 1
            right = (current * 2) + 2

            if num <= self.tree[left]:
                current = left
            else:
                num -= self.tree[left]
                current = right
            
            if current >= self.buffer_size - 1:
                break

        return self.tree[current], current, current - self.buffer_size + 1

    def update_priority(self, priority: float, index: int) -> None:
        """
        Update the priority of a data point at the given index.
        
        Args:
            priority (float): New priority value
            index (int): Index of the data point to update
        """
        self.tree[index] = priority
        self.__update_tree(index)

    def sum_all_priority(self) -> float:
        """
        Get the sum of all priorities in the tree.
        
        Returns:
            float: Sum of all priorities
        """
        return float(self.tree[0])


class MinTree:
    """
    A MinTree maintains a binary tree where each node represents the minimum value of its children.
    This is useful for quickly retrieving the minimum priority value in prioritized replay.
    
    Attributes:
        buffer_size (int): Maximum number of data points that can be stored
        tree (np.ndarray): Array representing the binary tree structure
        index (int): Current position in the tree for adding new data
    """
    
    def __init__(self, buffer_size: int):
        """
        Initialize the MinTree.
        
        Args:
            buffer_size (int): Maximum number of data points that can be stored
        """
        self.buffer_size = buffer_size
        self.tree = np.ones((buffer_size * 2 - 1))
        self.index = buffer_size - 1

    def __update_tree(self, index: int) -> None:
        """
        Update the tree values by propagating minimum values up from the given index.
        
        Args:
            index (int): Index to start updating from
        """
        while True:
            index = (index - 1) // 2
            left = (index * 2) + 1
            right = (index * 2) + 2
            if self.tree[left] > self.tree[right]:
                self.tree[index] = self.tree[right]
            else:
                self.tree[index] = self.tree[left]
            if index == 0:
                break

    def add_data(self, priority: float) -> None:
        """
        Add a new data point with the given priority.
        
        Args:
            priority (float): Priority value for the new data point
        """
        if self.index == self.buffer_size * 2 - 1:
            self.index = self.buffer_size - 1

        self.tree[self.index] = priority
        self.__update_tree(self.index)
        self.index += 1

    def update_priority(self, priority: float, index: int) -> None:
        """
        Update the priority of a data point at the given index.
        
        Args:
            priority (float): New priority value
            index (int): Index of the data point to update
        """
        self.tree[index] = priority
        self.__update_tree(index)

    def min_priority(self) -> float:
        """
        Get the minimum priority value in the tree.
        
        Returns:
            float: Minimum priority value
        """
        return float(self.tree[0])