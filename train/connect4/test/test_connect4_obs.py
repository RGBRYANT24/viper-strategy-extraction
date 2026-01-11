import sys
import os
# Add project root to python path to allow importing gym_env
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
# Add train/connect4 directory to python path to allow importing train_connect4_ppo_delta_selfplay
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import gym_env  # Import gym_env to register environments
from gym_env.policies.connect4_policies import Connect4MinMaxPolicy, Connect4RandomPolicy
from train_connect4_ppo_delta_selfplay import mask_fn_connect4


def test_observation():
    opponent_type = 'random'
    try:
        env = gym.make('Connect4-v0', opponent_type=opponent_type)
    except Exception as e:
        print(f"Error creating environment: {e}")
        return
    obs, _ = env.reset()
    # assert obs.shape == (6, 7)
    # assert np.all(obs == 0)
    env.render()
    env.close()

def test_input_observations():
    opponent_type = 'random'
    try:
        env = gym.make('Connect4-v0', opponent_type=opponent_type)
    except Exception as e:
        print(f"Error creating environment: {e}")
        return
    obs, _ = env.reset()
    env.render()

    print(env.unwrapped.board)

    # if hasattr(env, 'board'):
    #     board = env.board
    #     print('env hasattr board')
    #     print(board)
    # else:# 如果是包装过的环境
    #     board = env.env.board
    #     print('env.env hasattr board')
    #     print(board)

    if hasattr(env.unwrapped, 'board'):
        board = env.unwrapped.board
        print('env.unwrapped hasattr board')
        print(board)

    board[-1,1] = 1
    # env.render()
    print(board)
    mask = mask_fn_connect4(env)
    print('mask', mask)
    env.close()




if __name__ == '__main__':
    test_input_observations()


