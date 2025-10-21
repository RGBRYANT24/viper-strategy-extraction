"""
检查VIPER训练数据的格式
"""
import warnings
import numpy as np
from stable_baselines3 import DQN
from gym_env import make_env
from model.paths import get_oracle_path
import argparse

# 创建参数
args = argparse.Namespace(
    env_name='TicTacToe-v0',
    n_env=8,
    seed=42,
    verbose=0,
    tictactoe_opponent='minmax',
    tictactoe_minmax_depth=9,
    oracle_path='log/oracle_TicTacToe-v0_minmax.zip',
    total_timesteps=100000,
    n_iter=80,
    max_leaves=None,
    max_depth=None,
    log_prefix='',
    render=False
)

# 加载环境和Oracle
print("加载环境和Oracle...")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    env = make_env(args)
    oracle = DQN.load(get_oracle_path(args), env=env)
    env = oracle.env

print(f"环境类型: {type(env)}")
print(f"是否向量化: {hasattr(env, 'num_envs')}")
if hasattr(env, 'num_envs'):
    print(f"并行环境数: {env.num_envs}")

# 测试reset和step
print("\n测试环境输出...")
obs = env.reset()
print(f"Reset输出类型: {type(obs)}")
print(f"Reset输出shape: {obs.shape}")
print(f"Reset输出:\n{obs}")

# 预测动作
action, _ = oracle.predict(obs, deterministic=True)
print(f"\nOracle预测:")
print(f"  动作类型: {type(action)}")
print(f"  动作shape: {action.shape if hasattr(action, 'shape') else 'N/A'}")
print(f"  动作值: {action}")

# Step
next_obs, reward, done, info = env.step(action)
print(f"\nStep输出:")
print(f"  next_obs shape: {next_obs.shape}")
print(f"  reward shape: {reward.shape if hasattr(reward, 'shape') else type(reward)}")
print(f"  done shape: {done.shape if hasattr(done, 'shape') else type(done)}")

# 测试zip的行为
print("\n" + "="*70)
print("测试zip行为")
print("="*70)
print(f"obs shape: {obs.shape}")
print(f"action shape: {action.shape if hasattr(action, 'shape') else 'N/A'}")

# 模拟VIPER代码
trajectory_wrong = list(zip(obs, action, [1.0] * len(action)))
print(f"\n错误方式 - list(zip(obs, action, loss)):")
print(f"  trajectory长度: {len(trajectory_wrong)}")
print(f"  第一个元素类型: {type(trajectory_wrong[0])}")
print(f"  第一个元素: {trajectory_wrong[0]}")
print(f"  第一个观察维度: {trajectory_wrong[0][0].shape if hasattr(trajectory_wrong[0][0], 'shape') else 'N/A'}")

# 正确方式
trajectory_correct = [(obs[i], action[i], 1.0) for i in range(len(action))]
print(f"\n正确方式 - [(obs[i], action[i], loss) for i in range(len)]:")
print(f"  trajectory长度: {len(trajectory_correct)}")
print(f"  第一个元素: {trajectory_correct[0]}")
print(f"  第一个观察shape: {trajectory_correct[0][0].shape}")
print(f"  第一个动作: {trajectory_correct[0][1]}")
