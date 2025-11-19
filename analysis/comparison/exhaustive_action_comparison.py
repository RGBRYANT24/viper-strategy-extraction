"""
Exhaustive Action Comparison
Enumerates all legal Tic-Tac-Toe states (where it's X's turn) and compares
actions from Decision Tree, PPO, and MinMax.

NOTE: This script is designed to run on a server with the VIPER environment.
It uses the MinMax policy from `gym_env.policies.baseline_policies`.
It is self-contained and does NOT depend on `evaluation/battle_nn_vs_tree.py`.
"""

import os
import sys
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
import joblib
import gymnasium as gym
from gymnasium import spaces

# Add project root to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# Import MinMax from gym_env
from gym_env.policies.baseline_policies import MinMaxPlayerPolicy

# --- Embedded Player Classes to avoid dependencies ---

class LocalDecisionTreePlayer:
    """Decision Tree Player (Self-contained)"""
    def __init__(self, model_path):
        print(f"Loading Decision Tree from {model_path}...")
        self.model = joblib.load(model_path)
        
    def predict(self, obs):
        """Predict action for the given observation."""
        # Reshape for sklearn (1, n_features)
        obs_reshaped = obs.reshape(1, -1)
        
        # Get legal actions (indices where board is 0)
        legal_actions = np.where(obs == 0)[0]
        if len(legal_actions) == 0:
            return 0 # Should not happen in this script
        
        if hasattr(self.model, 'predict_proba'):
             # Use probability masking
             probs = self.model.predict_proba(obs_reshaped)[0]
             
             # Create mask: set illegal actions to -inf (or 0 probability)
             masked_probs = np.full(probs.shape, -np.inf)
             masked_probs[legal_actions] = probs[legal_actions]
             
             action = np.argmax(masked_probs)
        else:
            # Direct prediction
            prediction = self.model.predict(obs_reshaped)[0]
            
            # Handle vector outputs (e.g. Q-values from regressor)
            if np.size(prediction) > 1:
                # Create mask: set illegal actions to -inf
                masked_prediction = np.full(prediction.shape, -np.inf)
                masked_prediction[legal_actions] = prediction[legal_actions]
                
                action = np.argmax(masked_prediction)
            else:
                # Scalar output - cannot mask easily if it returns an illegal action directly
                # But we can check if it's legal, if not, we might have to fallback or keep it
                # For now, we keep it, as we can't "mask" a scalar choice without a distribution
                action = prediction
                
        # Ensure scalar
        if hasattr(action, 'item'):
            action = action.item()
            
        return int(action)

class LocalNeuralNetPlayer:
    """Neural Network Player (Self-contained, supports PPO and MaskablePPO)"""
    def __init__(self, model_path):
        print(f"Loading Neural Network from {model_path}...")
        self.model = self._load_model(model_path)
        
    def _load_model(self, model_path):
        # Try importing stable_baselines3
        try:
            from stable_baselines3 import PPO, A2C, DQN
        except ImportError:
            print("Error: stable_baselines3 not installed.")
            raise

        # Try importing sb3_contrib for MaskablePPO
        global MaskablePPO
        MaskablePPO = None
        try:
            from sb3_contrib import MaskablePPO
        except ImportError:
            pass

        # Attempt to load with different classes
        # 1. Try MaskablePPO if available
        if MaskablePPO is not None:
            try:
                model = MaskablePPO.load(model_path)
                print("Loaded as MaskablePPO")
                return model
            except:
                pass
                
        # 2. Try PPO
        try:
            model = PPO.load(model_path)
            print("Loaded as PPO")
            return model
        except:
            pass
            
        # 3. Try DQN
        try:
            model = DQN.load(model_path)
            print("Loaded as DQN")
            return model
        except:
            pass
            
        # 4. Try A2C
        try:
            model = A2C.load(model_path)
            print("Loaded as A2C")
            return model
        except:
            pass
            
        raise ValueError(f"Could not load model from {model_path} with any known class.")

    def predict(self, obs):
        """Predict action."""
        # Calculate action masks for MaskablePPO
        # Mask is True for valid actions, False for invalid
        # Board: 0 is empty (valid)
        action_masks = (obs == 0)
        
        # Check if model supports action_masks (MaskablePPO)
        if MaskablePPO is not None and isinstance(self.model, MaskablePPO):
            action, _ = self.model.predict(obs, deterministic=True, action_masks=action_masks)
        else:
            # Standard models don't support mask in predict
            action, _ = self.model.predict(obs, deterministic=True)
        
        # Ensure scalar
        if hasattr(action, 'item'):
            action = action.item()
            
        return int(action)


# --- Main Logic ---

def check_winner(board):
    """Check if there is a winner on the board."""
    win_combinations = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Cols
        [0, 4, 8], [2, 4, 6]              # Diagonals
    ]
    for combo in win_combinations:
        if abs(sum(board[combo])) == 3:
            return np.sign(sum(board[combo]))
    return 0

def generate_legal_states():
    """
    Generate all legal reachable states using BFS.
    Returns a list of states where it is Player 1's (X) turn.
    """
    states = []
    queue = [(np.zeros(9, dtype=int), 1)] # (board, current_player)
    visited = set()
    
    # Add empty board
    visited.add(tuple(np.zeros(9, dtype=int)))
    
    print("Generating all legal states...")
    
    while queue:
        board, player = queue.pop(0)
        board_tuple = tuple(board)
        
        # Check if game is over
        winner = check_winner(board)
        if winner != 0 or np.all(board != 0):
            continue
            
        # If it's Player 1's turn, add to our list
        if player == 1:
            states.append(board.copy())
            
        # Generate next states
        empty_indices = np.where(board == 0)[0]
        for idx in empty_indices:
            new_board = board.copy()
            new_board[idx] = player
            new_tuple = tuple(new_board)
            
            if new_tuple not in visited:
                visited.add(new_tuple)
                queue.append((new_board, -player))
                
    print(f"Found {len(states)} legal states where it is Player 1's turn.")
    return states

def board_to_string(board):
    """Convert board to a readable string format."""
    symbols = {1: 'X', -1: 'O', 0: '.'}
    s = "".join([symbols[x] for x in board])
    return f"{s[:3]}|{s[3:6]}|{s[6:]}"

def compare_actions(states, tree_player, nn_player, minmax_policy):
    """
    Compare actions of the three players on the given states.
    """
    results = []
    
    print("Comparing actions...")
    for board in tqdm(states):
        # MinMax (Ground Truth)
        minmax_action, _ = minmax_policy.predict(board, deterministic=True)
        
        # Decision Tree
        tree_action = tree_player.predict(board)
        
        # PPO
        nn_action = nn_player.predict(board)
        
        # Ensure actions are scalars (integers)
        minmax_action = int(np.array(minmax_action).item())
        tree_action = int(np.array(tree_action).item())
        nn_action = int(np.array(nn_action).item())
        
        # Check for differences
        diff_tree_minmax = tree_action != minmax_action
        diff_tree_nn = tree_action != nn_action
        diff_nn_minmax = nn_action != minmax_action
        
        # Record if there is ANY difference
        if diff_tree_minmax or diff_tree_nn or diff_nn_minmax:
            results.append({
                'Board': board_to_string(board),
                'Raw_Board': str(list(board.astype(int))),
                'Tree_Action': tree_action,
                'PPO_Action': nn_action,
                'MinMax_Action': minmax_action,
                'Diff_Tree_MinMax': diff_tree_minmax,
                'Diff_Tree_PPO': diff_tree_nn,
                'Diff_PPO_MinMax': diff_nn_minmax
            })
            
    return pd.DataFrame(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exhaustive Action Comparison")
    parser.add_argument("--viper-path", type=str, required=True, help="Path to VIPER decision tree model")
    parser.add_argument("--oracle-path", type=str, default="log/oracle_TicTacToe_ppo_aggressive.zip", help="Path to Oracle PPO model")
    parser.add_argument("--output", type=str, default="comparison_results.csv", help="Output CSV file")
    
    args = parser.parse_args()
    
    # Load models
    print("Loading models...")
    try:
        tree_player = LocalDecisionTreePlayer(args.viper_path)
        nn_player = LocalNeuralNetPlayer(args.oracle_path)
        
        # Initialize MinMax Policy from gym_env
        obs_space = spaces.Box(low=-1, high=1, shape=(9,), dtype=np.float32)
        act_space = spaces.Discrete(9)
        minmax_policy = MinMaxPlayerPolicy(obs_space, act_space, depth=9)
        
    except Exception as e:
        print(f"Error loading models: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
        
    # Generate states
    states = generate_legal_states()
    
    # Compare
    df = compare_actions(states, tree_player, nn_player, minmax_policy)
    
    # Save results
    if not df.empty:
        print(f"\nFound {len(df)} states with action differences.")
        df.to_csv(args.output, index=False)
        print(f"Saved results to {args.output}")
        
        # Print summary
        print("\nSummary of Differences:")
        print(f"Tree != MinMax: {df['Diff_Tree_MinMax'].sum()} states")
        print(f"Tree != PPO:    {df['Diff_Tree_PPO'].sum()} states")
        print(f"PPO != MinMax:  {df['Diff_PPO_MinMax'].sum()} states")
        
        print("\nSample Differences (Top 5):")
        print(df.head(5).to_string(index=False))
    else:
        print("\nNo differences found! The models are perfectly consistent on all legal states.")
