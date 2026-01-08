import gymnasium as gym
import numpy as np
import torch
import joblib
import os
import sys
import os
import argparse
from tqdm import tqdm
from sb3_contrib import MaskablePPO

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from gym_env.tictactoe import TicTacToeEnv

def get_state_key(board):
    return tuple(board.flatten())

def get_all_reachable_states():
    """
    Explore all unique reachable states in Tic-Tac-Toe (BFS).
    Returns a list of board inputs (numpy arrays).
    """
    visited = set()
    states_list = []
    # We need to know whose turn it is for the board.
    # Empty = X turn. 1 move = O turn.
    
    # Problem: The Oracle expects the observation, which includes the board.
    # In TicTacToeEnv, observation is simply the board (and maybe channel for player).
    # Let's check `TicTacToeEnv`.
    
    # Assuming obs is just board for now based on previous files.
    
    # Better approach: Recursive DFS
    
    valid_states = []
    
    def dfs(board, turn):
        key = tuple(board.flatten())
        if key in visited:
            return
        visited.add(key)
        
        # Check if terminal
        # simple check using the Env logic would be cleaner, but let's implement basic win check or use env
        # Instantiating env for every check is slow.
        
        # Let's store this state
        valid_states.append(board.copy())
        
        # Check for win/draw
        if check_win(board) or np.all(board != 0):
            return

        # Generate next states
        # Turn: 1 for X, -1 for O
        player = 1 if turn % 2 == 0 else -1
        
        # We only care about states where it's the AGENT'S turn if we are evaluating the agent?
        # Or do we want ALL states?
        # Usually Value function is V(s) for the current player. PPO handles perspective.
        
        empty_cells = np.where(board == 0)
        # Zip row, col
        moves = list(zip(empty_cells[0], empty_cells[1]))
        
        for r, c in moves:
            new_board = board.copy()
            new_board[r, c] = player
            dfs(new_board, turn + 1)

    dfs(np.zeros((3,3), dtype=int), 0)
    print(f"Found {len(valid_states)} reachable states.")
    return valid_states

def check_win(board):
    # Quick check
    for i in range(3):
        if abs(sum(board[i, :])) == 3: return True
        if abs(sum(board[:, i])) == 3: return True
    if abs(board.trace()) == 3: return True
    if abs(np.fliplr(board).trace()) == 3: return True
    return False

def generate_dataset(oracle_path, output_path):
    print(f"Loading Oracle from {oracle_path}...")
    env = gym.make('TicTacToe-v0')
    oracle = MaskablePPO.load(oracle_path, env=env)
    
    print("Generating reachable states...")
    raw_states = get_all_reachable_states()
    
    dataset = {
        'states': [],
        'values': [],
        'action_probs': []
    }
    
    print("Labeling data with Oracle...")
    with torch.no_grad():
        for board in tqdm(raw_states):
            # Evaluate only valid non-terminal states? Or all?
            # V(terminal) should be outcome. Oracle might give weird values for terminal states if not trained well,
            # but usually it's fine.
            # However, `predict_values` expects observation.
            
            # Note: The Env might flip the board for O's turn if using self-play wrappers.
            # If we just pass the raw board to the PPO trained on 'minmax' (which usually sees canonical board),
            # we need to be careful.
            # Assuming 'oracle_TicTacToe_ppo_aggressive.zip' expects the board as is relative to 'current player'.
            # Our BFS generates boards with 1 and -1.
            # If it's O's turn (-1), the board seen by the agent should be inverted if the agent always plays as '1'.
            # Let's inspect `TicTacToeEnv` later if needed. For now, we assume standard observing.
            
            obs = board # In v0 env, obs is the board
            
            # Prepare tensor
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(oracle.device)
            
            # 1. Get Value
            value = oracle.policy.predict_values(obs_tensor).item()
            
            # 2. Get Action Probabilities
            # MaskablePPO logic for probabilities
            # We need action masks to get valid probabilities
            mask = (obs.flatten() == 0)
            if not np.any(mask): # Full board
                action_probs = np.zeros(9)
            else:
                # Get distribution
                # sb3_contrib.common.maskable.policies.MaskableActorCriticPolicy
                # forward(obs, action_masks) -> features -> distribution
                
                # We can use `get_distribution`
                mask_tensor = torch.as_tensor(mask, dtype=torch.bool).unsqueeze(0).to(oracle.device)
                dist = oracle.policy.get_distribution(obs_tensor, action_masks=mask_tensor)
                probs = dist.distribution.probs.cpu().numpy()[0] # Categorical distribution
                action_probs = probs

            dataset['states'].append(obs)
            dataset['values'].append(value)
            dataset['action_probs'].append(action_probs)
            
    # Convert to arrays
    dataset['states'] = np.array(dataset['states'])
    dataset['values'] = np.array(dataset['values'])
    dataset['action_probs'] = np.array(dataset['action_probs'])
    
    print(f"Dataset generated: {len(dataset['states'])} samples.")
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(dataset, output_path)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--oracle-path', type=str, required=True)
    parser.add_argument('--output_path', type=str, default='data/tictactoe_oracle_dataset.joblib')
    args = parser.parse_args()
    
    generate_dataset(args.oracle_path, args.output_path)
