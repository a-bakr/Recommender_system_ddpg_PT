# Deep Reinforcement Learning based Recommender System

This project implements a Deep Reinforcement Learning based Recommender System based on the paper [Deep Reinforcement Learning based Recommendation with Explicit User-Item Interactions Modeling](https://arxiv.org/abs/1810.12027) by Liu et al. The system uses the DDPG (Deep Deterministic Policy Gradient) algorithm and incorporates multimodal features for enhanced recommendations.

## Project Overview

The recommender system combines traditional collaborative filtering with deep reinforcement learning and multimodal features to provide personalized movie recommendations. It uses:
- DDPG algorithm for reinforcement learning
- Multimodal features (video, audio, text) for rich item representation
- State representation module for RL algorithm
- User-item interaction modeling

## Project Structure

```
.
├── data/               # Dataset and processed features
├── src/               # Source code
│   ├── embedding.py   # Embedding models
│   └── ...
├── scripts/           # Training and evaluation scripts
├── save_model/        # Saved model checkpoints
├── save_weights/      # Saved model weights
├── train.py          # Training script
├── eval.py           # Evaluation script
└── run.py            # Main execution script
```

## Features

### Multimodal Embedding
The system incorporates multimodal movie features using:
- Visual features (processed by ViT)
- Audio features (processed by AST)
- Text features (processed by BERT)

Two fusion strategies are implemented:
1. **Early Fusion**: Single FC layer applied to pooled multimodal features
2. **Late Fusion**: Modality-specific FC layers applied before pooling

Pooling methods:
- Concatenation
- Element-wise mean

## Installation

1. Create and activate conda environment:
```bash
conda create -n env_name python=3.11.2 
conda activate env_name
```

2. Install dependencies:
```bash
# Install TensorFlow with CUDA support
pip install tensorflow[and-cuda]
conda install tensorflow==2.12.0

# Install PyTorch with CUDA support
pip install torch==2.0.1+cu117 --index-url https://download.pytorch.org/whl/cu117

# Install other requirements
pip install -r requirements.txt
```

## Dataset

The project uses the [MovieLens 1M Dataset](https://grouplens.org/datasets/movielens/1m/). To prepare the dataset:

```bash
unzip ./ml-1m.zip
```

## Usage

### Training and Evaluation

1. Run with multimodal features (embedding dimension = 128):
```bash
bash scripts/train_modality.sh
bash scripts/eval_modality.sh
```

2. Run with single ID features (embedding dimension = 100):
```bash
bash scripts/train.sh
bash scripts/eval.sh
```

## Model Architecture

The system consists of several key components:

1. **Embedding Models**:
   - MovieGenreEmbedding: Handles movie-genre relationships
   - UserMovieEmbedding: Manages user-movie interactions
   - UserMovieMultiModalEmbedding: Incorporates multimodal features

2. **State Representation**:
   - Processes user-item interactions
   - Generates trainable states for RL algorithm

3. **DDPG Algorithm**:
   - Actor-Critic architecture
   - Continuous action space
   - Experience replay buffer

## References

- Original paper: [Deep Reinforcement Learning based Recommendation with Explicit User-Item Interactions Modeling](https://arxiv.org/abs/1810.12027)
- DDPG paper: [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)
- Base implementation reference: [Recommender_system_via_deep_RL](https://github.com/backgom2357/Recommender_system_via_deep_RL)
