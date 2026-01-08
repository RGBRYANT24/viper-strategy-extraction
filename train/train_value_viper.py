import numpy as np
import torch
import gymnasium as gym
from sklearn.tree import DecisionTreeRegressor
from sb3_contrib import MaskablePPO
import joblib
import sys
import os
import argparse
from datetime import datetime

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gym_env.tictactoe import TicTacToeEnv

def get_oracle_value(oracle, obs):
    """
    Extract the Value Function V(s) from the PPO Critic.
    
    Args:
        oracle: MaskablePPO model
        obs: Observation (numpy array)
        
    Returns:
        value: Scalar value estimate for the state
    """
    with torch.no_grad():
        # Convert observation to tensor
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(oracle.device)
        
        # In SB3, predict_values returns the value estimate from the critic
        # shape: (1, 1) -> scalar
        value = oracle.policy.predict_values(obs_tensor)
        return value.item()

def sample_trajectory_value(oracle, env, n_steps):
    """
    Sample trajectories using the Oracle policy and record (s, V(s)).
    
    Args:
        oracle: PPO model (acting as teacher)
        env: Gym environment
        n_steps: Number of steps to sample
        
    Returns:
        dataset: List of (obs, value) tuples
    """
    dataset = []
    obs, _ = env.reset()
    
    for _ in range(n_steps):
        # 1. Get Action from Oracle (Policy)
        mask = (obs == 0).astype(bool)
        mask_tensor = torch.tensor(mask).unsqueeze(0)
        action, _ = oracle.predict(obs, deterministic=True, action_masks=mask_tensor)
        
        # 2. Get Value from Oracle (Critic)
        value = get_oracle_value(oracle, obs)
        
        # 3. Store data (s, V(s))
        # Note: We store the current observation and its estimated value
        dataset.append((obs.copy(), value))
        
        # 4. Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        if done:
            obs, _ = env.reset()
            
    return dataset

def train_value_tree(dataset, max_depth=10, max_leaves=50, min_samples_split=10, min_samples_leaf=5):
    """
    Train a Decision Tree Regressor to approximate V(s).
    """
    X = np.array([state for state, _ in dataset])
    y = np.array([value for _, value in dataset])
    
    print(f"Training Data: X.shape={X.shape}, y.shape={y.shape}")
    print(f"Value Range: min={y.min():.3f}, max={y.max():.3f}, mean={y.mean():.3f}")

    tree = DecisionTreeRegressor(
        max_depth=max_depth,
        max_leaf_nodes=max_leaves,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    
    tree.fit(X, y)
    
    print(f"✓ Training Complete")
    print(f"  Depth: {tree.tree_.max_depth}")
    print(f"  Leaves: {tree.tree_.n_leaves}")
    
    return tree

def evaluate_value_tree(tree, oracle, env, n_samples=1000):
    """
    Evaluate the Value Tree by comparing predictions with Oracle's V(s) on new samples.
    """
    print(f"\nEvaluating Value Tree on {n_samples} new samples...")
    dataset = sample_trajectory_value(oracle, env, n_samples)
    
    X_test = np.array([state for state, _ in dataset])
    y_true = np.array([value for _, value in dataset])
    
    y_pred = tree.predict(X_test)
    
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = tree.score(X_test, y_true)
    
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  R2 Score: {r2:.6f}")
    
    return mse, r2

def train_value_viper_loop(oracle_path, output_path, n_iterations, samples_per_iter, 
                           max_depth, max_leaves, min_samples_split, min_samples_leaf):
    
    # Setup paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not output_path:
        leaves_str = str(max_leaves) if max_leaves is not None else "all-leaves"
        depth_str = str(max_depth) if max_depth is not None else "all-depth"
        # Using a more descriptive folder name for the critic/value function trees
        output_path = f"log/value_distillation/value_viper_TicTacToe-v0_{leaves_str}_{depth_str}_{timestamp}.joblib"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load Oracle
    print(f"Loading Oracle from {oracle_path}...")
    # Fix: PPO might require specific env params, but for loading just needs correct shape usually.
    # We create a dummy env.
    env = gym.make('TicTacToe-v0', opponent_type='random')
    oracle = MaskablePPO.load(oracle_path, env=env)
    
    all_data = []
    best_tree = None
    best_mse = float('inf')
    
    for i in range(n_iterations):
        print(f"\n{'='*60}")
        print(f"Iteration {i+1}/{n_iterations}")
        print(f"{'='*60}")
        
        # 1. Sample
        new_data = sample_trajectory_value(oracle, env, samples_per_iter)
        all_data.extend(new_data)
        
        # 2. Train
        tree = train_value_tree(
            all_data, 
            max_depth=max_depth, 
            max_leaves=max_leaves,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf
        )
        
        # 3. Evaluate
        mse, r2 = evaluate_value_tree(tree, oracle, env, n_samples=2000)
        
        if mse < best_mse:
            best_mse = mse
            best_tree = tree
            print(f"  -> New Best Tree found!")

    # Save Best Tree
    print(f"\nSaving best value tree to {output_path}...")
    joblib.dump(best_tree, output_path)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Value Function Regression Tree (V(s)) from PPO Oracle")
    parser.add_argument('--oracle_path', type=str, default='log/oracle_TicTacToe_ppo_aggressive.zip',
                        help='Path to the PPO oracle model')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Path to save the trained value tree')
    parser.add_argument('--n_iterations', type=int, default=5,
                        help='Number of DAgger-like iterations (sampling + training)')
    parser.add_argument('--samples_per_iter', type=int, default=5000,
                        help='Number of samples to collect per iteration')
    parser.add_argument('--max_depth', type=int, default=10,
                        help='Max depth of the tree')
    parser.add_argument('--max_leaves', type=int, default=100,
                        help='Max leaf nodes in the tree')
    parser.add_argument('--min_samples_split', type=int, default=20,
                        help='Min samples required to split an internal node')
    parser.add_argument('--min_samples_leaf', type=int, default=10,
                        help='Min samples required to be at a leaf node')

    args = parser.parse_args()
    
    train_value_viper_loop(
        oracle_path=args.oracle_path,
        output_path=args.output_path,
        n_iterations=args.n_iterations,
        samples_per_iter=args.samples_per_iter,
        max_depth=args.max_depth,
        max_leaves=args.max_leaves,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf
    )
