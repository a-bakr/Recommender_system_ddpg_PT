"""
Embedding models for the recommender system.

This module implements various embedding models for the recommender system:
1. MovieGenreEmbedding: Handles movie-genre relationships
2. UserMovieEmbedding: Manages user-movie interactions
3. UserMovieMultiModalEmbedding: Incorporates multimodal features
4. ConcatedEmbedding: Combines different embedding networks
"""

import torch
import torch.nn as nn
import tensorflow as tf
import os 
import numpy as np

class MovieGenreEmbedding(nn.Module):
    """
    PyTorch model for embedding movie-genre relationships.
    
    This model learns embeddings for movies and genres, and computes their similarity
    using cosine similarity. The output is passed through a fully connected layer
    and sigmoid activation to predict movie-genre relationships.
    
    Args:
        len_movies (int): Number of unique movies
        len_genres (int): Number of unique genres
        embedding_dim (int): Dimension of the embedding vectors
    """
    def __init__(self, len_movies, len_genres, embedding_dim):
        super(MovieGenreEmbedding, self).__init__()
        
        # Embedding layers : trainable
        self.m_embedding = nn.Embedding(num_embeddings=len_movies, embedding_dim=embedding_dim)
        self.g_embedding = nn.Embedding(num_embeddings=len_genres, embedding_dim=embedding_dim)
        
        # Dot product layer
        self.m_g_merge = nn.CosineSimilarity(dim=1)
        
        # Output layer
        self.m_g_fc = nn.Linear(1, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 2) containing movie and genre IDs
            
        Returns:
            torch.Tensor: Predicted movie-genre relationship scores
        """
        memb = self.m_embedding(x[:, 0])
        gemb = self.g_embedding(x[:, 1])
        m_g = self.m_g_merge(memb, gemb).unsqueeze(1)
        
        return self.sigmoid(self.m_g_fc(m_g))

class UserMovieEmbedding(nn.Module):
    """
    PyTorch model for embedding user-movie interactions.
    
    This model implements a matrix factorization approach using embeddings for users
    and movies. It computes the dot product of user and movie embeddings and passes
    the result through a fully connected layer with sigmoid activation.
    
    Args:
        len_users (int): Number of unique users
        len_movies (int): Number of unique movies
        embedding_dim (int): Dimension of the embedding vectors
    """
    def __init__(self, len_users, len_movies, embedding_dim):
        super(UserMovieEmbedding, self).__init__()
        
        # Embedding layers : trainable
        self.u_embedding = nn.Embedding(num_embeddings=len_users, embedding_dim=embedding_dim)
        self.m_embedding = nn.Embedding(num_embeddings=len_movies, embedding_dim=embedding_dim)
        
        # Dot product layer
        self.m_u_merge = nn.CosineSimilarity(dim=1)
        
        # Output layer
        self.m_u_fc = nn.Linear(1, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 2) containing user and movie IDs
            
        Returns:
            torch.Tensor: Predicted user-movie interaction scores
        """
        user_ids, movie_ids = x[:, 0], x[:, 1]
        
        uemb = self.u_embedding(user_ids)
        memb = self.m_embedding(movie_ids)
        m_u = (uemb * memb).sum(dim=1, keepdim=True) # Matrix Factorization Model
        
        return torch.sigmoid(self.m_u_fc(m_u))

class UserMovieMultiModalEmbedding(tf.keras.Model):
    """
    TensorFlow model for multimodal user-movie embeddings.
    
    This model incorporates multimodal features (video, audio, text) for movies
    along with user embeddings. It supports different fusion strategies (early/late)
    and aggregation methods (concatenation/mean).
    
    Args:
        len_users (int): Number of unique users
        len_movies (int): Number of unique movies
        embedding_dim (int): Dimension of the embedding vectors
        modality (tuple): Tuple of modalities to use ('video', 'audio', 'text')
        fusion (str): Fusion strategy ('early' or 'late')
        aggregation (str): Aggregation method ('concat' or 'mean')
    """
    def __init__(self, 
                    len_users, 
                    len_movies, 
                    embedding_dim, 
                    modality=('video', 'audio', 'text'), 
                    fusion='early', 
                    aggregation='concat'):
        
        super(UserMovieMultiModalEmbedding, self).__init__()
        self.modality = modality
        self.fusion = fusion
        self.aggregation = aggregation
        
        # input: (user, movie)
        self.m_u_input = tf.keras.layers.InputLayer(name='input_layer', input_shape=(2,))
        
        # user embedding
        self.u_embedding = tf.keras.layers.Embedding(name='user_embedding', input_dim=len_users, output_dim=embedding_dim)
        
        # item embedding        
        if not modality:
            self.m_embedding = tf.keras.layers.Embedding(name='movie_embedding', input_dim=len_movies, output_dim=embedding_dim)
        
        else:
            # load multimodal features
            for mod in modality:
                ROOT_DIR = os.getcwd()
                DATA_DIR = os.path.join(ROOT_DIR, 'data/ml-1m')
                mod_name = 'image' if mod == 'video' else mod # rename due to file name
                setattr(self, f'{mod}_feat', np.load(f'{DATA_DIR}/{mod_name}_feat.npy'))
                
            if fusion == 'early':
                self.mm_fc = tf.keras.layers.Dense(embedding_dim, name='mm_fc')
                
            elif fusion == 'late':
                if aggregation == 'concat':
                    def divide_integer(n, parts):
                        q, r = divmod(n, parts)
                        return [q+1]*(r) + [q]*(parts-r)
                    embedding_dims = divide_integer(embedding_dim, len(modality))
                elif aggregation == 'mean':
                    embedding_dims = [embedding_dim]*len(modality)
                    
                for i, mod in enumerate(modality):
                    setattr(self, f'{mod}_fc', tf.keras.layers.Dense(embedding_dims[i], name=f'{mod}_fc'))
        
        # dot product
        self.m_u_merge = tf.keras.layers.Dot(name='movie_user_dot', normalize=False, axes=1)
        # output
        self.m_u_fc = tf.keras.layers.Dense(1, activation='sigmoid')
        
    def get_embedding(self, x):
        """
        Get user and movie embeddings.
        
        Args:
            x (tf.Tensor): Input tensor containing user and movie IDs
            
        Returns:
            tuple: (user_embedding, movie_embedding)
        """
        x = self.m_u_input(x)
        uemb = self.u_embedding(x[0])
        
        if not self.modality:
            memb = self.m_embedding(x[1])
        else:
            mm_emb = []
            for mod in self.modality:
                mm_feat = getattr(self, f'{mod}_feat')
                x[1] = tf.cast(x[1], tf.int32)
                x[0] = tf.cast(x[0], tf.int32)
                mm_feat = tf.gather(mm_feat, x[1])
                
                if self.fusion == 'early':
                    mm_emb.append(mm_feat)
                elif self.fusion == 'late':
                    mm_emb.append(getattr(self, f'{mod}_fc')(mm_feat))
            
            if self.aggregation == 'concat':
                memb = tf.concat(mm_emb, axis=1)
            elif self.aggregation == 'mean':
                memb = tf.reduce_mean(tf.stack(mm_emb), axis=0)
                
            if self.fusion == 'early':
                memb = self.mm_fc(memb)
        return uemb, memb
        
    def call(self, x):
        """
        Forward pass of the model.
        
        Args:
            x (tf.Tensor): Input tensor containing user and movie IDs
            
        Returns:
            tf.Tensor: Predicted user-movie interaction scores
        """
        uemb, memb = self.get_embedding(x)
        m_u = self.m_u_merge([memb, uemb])
        return self.m_u_fc(m_u)

class ConcatedEmbedding:
    """
    Combines different embedding networks.
    
    This class combines the outputs of an ID-based embedding network and a
    multimodal embedding network by concatenating their embeddings.
    
    Args:
        id_embedding_network: Network for ID-based embeddings
        mm_embedding_network: Network for multimodal embeddings
    """
    def __init__(self, id_embedding_network, mm_embedding_network):
        self.id_embedding_network = id_embedding_network
        self.mm_embedding_network = mm_embedding_network
        
    def get_embedding(self, x):
        """
        Get combined embeddings from both networks.
        
        Args:
            x (tf.Tensor): Input tensor containing user and movie IDs
            
        Returns:
            tuple: (combined_user_embedding, combined_movie_embedding)
        """
        id_uemb, id_memb = self.id_embedding_network.get_embedding(x)
        if self.mm_embedding_network:
            mm_uemb, mm_memb = self.mm_embedding_network.get_embedding(x)
        else:
            mm_uemb = tf.zeros_like(id_uemb)
            mm_memb = tf.zeros_like(id_memb)
        uemb, memb = tf.concat([id_uemb, mm_uemb], axis=0), tf.concat([id_memb, mm_memb], axis=1)
        return uemb, memb

if __name__ == "__main__":
    pass