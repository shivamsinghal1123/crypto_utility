"""
Reinforcement Learning agent with neural network for crypto price prediction.
Uses Deep Q-Learning with experience replay for optimal prediction strategy.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging
import json

logger = logging.getLogger(__name__)


class PolicyNetwork(nn.Module):
    """Neural network for Q-value approximation."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = None):
        """
        Initialize policy network.
        
        Args:
            state_dim: Input state dimension
            action_dim: Number of possible actions
            hidden_dims: Hidden layer dimensions
        """
        super(PolicyNetwork, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]
        
        layers = []
        in_dim = state_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            in_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(in_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
        logger.info(f"Policy network created: {state_dim} -> {hidden_dims} -> {action_dim}")
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through network.
        
        Args:
            state: Input state tensor
            
        Returns:
            Q-values for each action
        """
        return self.network(state)


class RLAgent:
    """
    Reinforcement Learning agent for trading predictions.
    Uses Deep Q-Learning with experience replay.
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 learning_rate: float = 0.001,
                 gamma: float = 0.95,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.05,
                 epsilon_decay: float = 0.995,
                 memory_size: int = 10000,
                 batch_size: int = 64,
                 model_path: str = None):
        """
        Initialize RL agent.
        
        Args:
            state_dim: State vector dimension
            action_dim: Number of possible actions
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for future rewards
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Exploration decay rate
            memory_size: Experience replay buffer size
            batch_size: Training batch size
            model_path: Path to save/load model
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        # Experience replay memory
        self.memory = deque(maxlen=memory_size)
        
        # Neural networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.target_net = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
        # Model persistence
        if model_path is None:
            model_path = Path(__file__).parent.parent / "data" / "rl_model.pth"
        self.model_path = Path(model_path)
        
        # Training metrics
        self.training_losses = []
        self.episode_rewards = []
        
        logger.info(f"RL Agent initialized on {self.device}")
        logger.info(f"State dim: {state_dim}, Action dim: {action_dim}")
        logger.info(f"Epsilon: {epsilon_start} -> {epsilon_end}, Decay: {epsilon_decay}")
    
    def select_action(self, state: np.ndarray, explore: bool = True) -> Tuple[int, float]:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state vector
            explore: Whether to use exploration
            
        Returns:
            (action_index, confidence)
        """
        # Exploration
        if explore and random.random() < self.epsilon:
            action = random.randrange(self.action_dim)
            confidence = 1.0 / self.action_dim  # Low confidence for random action
            logger.debug(f"Exploration: random action {action}")
        # Exploitation
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                action = q_values.argmax().item()
                
                # Confidence based on Q-value distribution
                q_probs = torch.softmax(q_values, dim=1)
                confidence = q_probs[0, action].item()
                
                logger.debug(f"Exploitation: action {action}, confidence {confidence:.2%}")
        
        return action, confidence
    
    def store_experience(self,
                        state: np.ndarray,
                        action: int,
                        reward: float,
                        next_state: np.ndarray,
                        done: bool):
        """
        Store experience in replay buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def train_step(self) -> Optional[float]:
        """
        Perform one training step on a batch from memory.
        
        Returns:
            Loss value or None if insufficient samples
        """
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Current Q-values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Target Q-values
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones.float()) * self.gamma * next_q
        
        # Calculate loss
        loss = self.loss_fn(current_q.squeeze(), target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        loss_value = loss.item()
        self.training_losses.append(loss_value)
        
        return loss_value
    
    def update_target_network(self):
        """Update target network with policy network weights."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
        logger.info("Target network updated")
    
    def train_episode(self, experiences: List[Tuple]) -> Dict:
        """
        Train on a full episode of experiences.
        
        Args:
            experiences: List of (state, action, reward, next_state, done) tuples
            
        Returns:
            Training metrics
        """
        # Add all experiences to memory
        for exp in experiences:
            self.store_experience(*exp)
        
        # Train multiple times
        losses = []
        for _ in range(len(experiences)):
            loss = self.train_step()
            if loss is not None:
                losses.append(loss)
        
        # Calculate episode reward
        episode_reward = sum(exp[2] for exp in experiences)
        self.episode_rewards.append(episode_reward)
        
        metrics = {
            'episode_reward': episode_reward,
            'avg_loss': np.mean(losses) if losses else 0,
            'epsilon': self.epsilon,
            'memory_size': len(self.memory)
        }
        
        logger.info(f"Episode trained: reward={episode_reward:.2f}, "
                   f"avg_loss={metrics['avg_loss']:.4f}, "
                   f"epsilon={self.epsilon:.3f}")
        
        return metrics
    
    def save_model(self):
        """Save model checkpoint."""
        checkpoint = {
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_losses': self.training_losses,
            'episode_rewards': self.episode_rewards
        }
        
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, self.model_path)
        logger.info(f"Model saved to {self.model_path}")
    
    def load_model(self) -> bool:
        """
        Load model checkpoint.
        
        Returns:
            True if loaded successfully
        """
        if not self.model_path.exists():
            logger.warning(f"No model found at {self.model_path}")
            return False
        
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint['epsilon']
            self.training_losses = checkpoint['training_losses']
            self.episode_rewards = checkpoint['episode_rewards']
            
            logger.info(f"Model loaded from {self.model_path}")
            logger.info(f"Epsilon: {self.epsilon:.3f}, "
                       f"Episodes: {len(self.episode_rewards)}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def get_performance_stats(self) -> Dict:
        """
        Get training performance statistics.
        
        Returns:
            Performance metrics
        """
        if not self.episode_rewards:
            return {
                'episodes': 0,
                'avg_reward': 0,
                'best_reward': 0,
                'recent_avg_reward': 0
            }
        
        recent_window = min(100, len(self.episode_rewards))
        
        return {
            'episodes': len(self.episode_rewards),
            'avg_reward': np.mean(self.episode_rewards),
            'best_reward': np.max(self.episode_rewards),
            'worst_reward': np.min(self.episode_rewards),
            'recent_avg_reward': np.mean(self.episode_rewards[-recent_window:]),
            'recent_best_reward': np.max(self.episode_rewards[-recent_window:]),
            'avg_loss': np.mean(self.training_losses[-1000:]) if self.training_losses else 0,
            'epsilon': self.epsilon
        }
