
import gymnasium as gym
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import gym_env
import numpy as np

# Ensure gym_env is imported to trigger registration

def test_connect4_basics():
    start_msg = "=== Testing Connect4 Environment Basics ==="
    print(start_msg)
    
    # Test 1: Random opponent
    env = gym.make('Connect4-v0', opponent_type='random')
    obs, info = env.reset(seed=42)
    print(f"Observation shape: {obs.shape}")
    print(f"Initial Board (Self=1, Opp=-1):\n{obs}")
    
    # Test step
    action = 0
    obs, reward, done, truncated, info = env.step(action)
    print(f"Step 1: Action {action}, Reward {reward}, Done {done}, Info {info}")
    env.render()
    
    # Loop until done
    for i in range(20):
        if done: break
        # pick valid action
        valid = [c for c in range(7) if obs[5][c] == 0] # checking top row in obs (row count=6 -> index 5)
        # Note: In our env, row 0 is usually bottom, row 5 is top. Logic check needed.
        # My env implementation:
        # _drop_piece iterates r from 0 to 5. if board[r][col]==0 -> fill.
        # So index 0 is bottom. Index 5 is top.
        # _is_valid_action checks board[ROW_COUNT-1][col] aka board[5][col] == 0.
        
        if not valid: break
        action = np.random.choice(valid)
        obs, reward, done, truncated, info = env.step(action)
        print(f"Step {i+2}: Action {action}, Reward {reward}, Done {done}")
        env.render()
        
    print(f"Game Over. Winner: {env.unwrapped.winner}")
    print("=== Connect4 Basics Test Passed ===\n")

def test_connect4_minmax():
    print("=== Testing Connect4 MinMax Opponent ===")
    env = gym.make('Connect4-v0', opponent_type='minmax', minmax_depth=2)
    obs, info = env.reset(seed=100)
    
    # Just run a few steps to ensure no crash
    done = False
    step = 0
    while not done and step < 5:
        # valid actions
        # obs is from my perspective. 0 is empty.
        valid = [c for c in range(7) if obs[5][c] == 0] 
        action = np.random.choice(valid)
        obs, reward, done, truncated, info = env.step(action)
        print(f"Step {step}: Action {action}, Reward {reward}")
        env.render()
        step += 1
        
    print("=== Connect4 MinMax Test Passed (No Crash) ===\n")

if __name__ == "__main__":
    test_connect4_basics()
    test_connect4_minmax()
