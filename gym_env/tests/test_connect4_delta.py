
import gymnasium as gym
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import gym_env
import numpy as np
from gym_env.policies.connect4_policies import Connect4MinMaxPolicy, Connect4RandomPolicy

def test_connect4_delta_selfplay():
    print("=== Testing Connect4 Delta-Uniform Self-Play Environment ===")
    
    # Create baseline pool
    obs_space = gym.spaces.Box(low=-1, high=1, shape=(6, 7), dtype=np.float32)
    act_space = gym.spaces.Discrete(7)
    
    baseline_pool = [
        Connect4RandomPolicy(obs_space, act_space),
        Connect4MinMaxPolicy(obs_space, act_space, depth=1) # shallow for speed
    ]
    
    env = gym.make('Connect4-DeltaSelfPlay-v0', 
                   baseline_pool=baseline_pool,
                   learned_pool=None, 
                   play_as_o_prob=0.5)
                   
    print("Environment created successfully.")
    
    # Test 1: Reset and check opponent sampling
    print("\n-- Test 1: Reset & Opponent Sampling --")
    obs, info = env.reset(seed=42)
    print(f"Initial Observation shape: {obs.shape}")
    print(f"Current Opponent: {env.unwrapped.current_opponent}")
    print(f"Playing as: {'O (Second)' if env.unwrapped.play_as_o else 'X (First)'}")
    env.render()
    
    # Test 2: Play a few steps
    print("\n-- Test 2: Play Loop --")
    
    done = False
    step = 0
    while not done and step < 10:
        valid = [c for c in range(7) if obs[5][c] == 0] # checking top row from my perspective? 
        # Wait, if obs is flipped, does 0 still mean empty? YES. 0 is always empty.
        # But if obs is flipped, my pieces are 1, opp are -1.
        
        if not valid: break
        action = np.random.choice(valid)
        
        obs, reward, done, truncated, info = env.step(action)
        print(f"Step {step}: Action {action}, Reward {reward}, Info {info}")
        env.render()
        step += 1
        
    print(f"Game Over. Winner: {env.unwrapped.winner}")
    print("=== Connect4 Delta Self-Play Test Passed ===\n")

if __name__ == "__main__":
    test_connect4_delta_selfplay()
