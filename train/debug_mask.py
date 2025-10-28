"""
调试脚本：检查 MaskablePPO 的 action masking 是否正常工作
"""

import gymnasium as gym
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gym_env

# 导入 PolicySnapshot
from train_delta_selfplay_ppo import PolicySnapshot


def mask_fn(env):
    """返回 action mask"""
    # 尝试多种方式获取 board
    if hasattr(env, 'board'):
        board = env.board
    elif hasattr(env, 'env') and hasattr(env.env, 'board'):
        board = env.env.board
    elif hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'board'):
        board = env.unwrapped.board
    else:
        print(f"⚠️  无法找到 board 属性！")
        print(f"   env 类型: {type(env)}")
        print(f"   env 属性: {dir(env)}")
        # 返回全 1 mask（所有动作合法）作为回退
        return np.ones(9, dtype=np.int8)

    mask = (board == 0).astype(np.int8)
    return mask


def test_mask_function():
    """测试 mask_fn 是否能正确访问环境"""
    print("=" * 70)
    print("测试 1: 检查 mask_fn 能否正确访问环境")
    print("=" * 70)

    # 创建环境
    env = gym.make('TicTacToe-v0', opponent_type='random')
    env = Monitor(env)
    env = ActionMasker(env, mask_fn)

    obs, _ = env.reset()
    print(f"✓ 环境创建成功")
    print(f"  环境类型: {type(env)}")
    print(f"  观察: {obs}")

    # 测试 mask_fn
    mask = mask_fn(env)
    print(f"✓ mask_fn 调用成功")
    print(f"  Mask: {mask}")
    print(f"  合法动作数: {mask.sum()}")

    # 检查环境的 board 访问路径
    print(f"\n检查环境结构:")
    print(f"  hasattr(env, 'board'): {hasattr(env, 'board')}")
    print(f"  hasattr(env, 'env'): {hasattr(env, 'env')}")
    if hasattr(env, 'env'):
        print(f"  hasattr(env.env, 'board'): {hasattr(env.env, 'board')}")
        print(f"  hasattr(env.env, 'env'): {hasattr(env.env, 'env')}")
        if hasattr(env.env, 'env'):
            print(f"  hasattr(env.env.env, 'board'): {hasattr(env.env.env, 'board')}")
    print(f"  hasattr(env, 'unwrapped'): {hasattr(env, 'unwrapped')}")
    if hasattr(env, 'unwrapped'):
        print(f"  hasattr(env.unwrapped, 'board'): {hasattr(env.unwrapped, 'board')}")

    env.close()
    print()


def test_maskable_ppo_prediction():
    """测试 MaskablePPO 的 predict 是否使用 mask"""
    print("=" * 70)
    print("测试 2: 检查 MaskablePPO.predict() 是否使用 mask")
    print("=" * 70)

    # 创建环境
    env = gym.make('TicTacToe-v0', opponent_type='random')
    env = Monitor(env)
    env = ActionMasker(env, mask_fn)

    # 创建一个新的模型（随机初始化）
    model = MaskablePPO('MlpPolicy', env, verbose=0)

    # 手动设置一个已占用的棋盘
    obs, _ = env.reset()
    print(f"初始观察: {obs}")

    # 手动修改棋盘（模拟已有棋子）
    if hasattr(env, 'unwrapped'):
        env.unwrapped.board[0] = 1   # 位置 0 被 X 占据
        env.unwrapped.board[4] = -1  # 位置 4 被 O 占据
        obs = env.unwrapped.board.copy()
        print(f"修改后棋盘: {obs}")

    # 测试 mask
    mask = mask_fn(env)
    print(f"Mask: {mask}")
    print(f"合法动作: {np.where(mask == 1)[0].tolist()}")
    print(f"非法动作: {np.where(mask == 0)[0].tolist()}")

    # 测试 predict（多次）
    print("\n测试 1: 不传递 action_masks（100次）:")
    illegal_count = 0
    action_counts = {i: 0 for i in range(9)}

    for _ in range(100):
        action, _ = model.predict(obs, deterministic=False)
        action_counts[int(action)] += 1
        if mask[action] == 0:
            illegal_count += 1

    print(f"  动作分布: {action_counts}")
    print(f"  非法动作次数: {illegal_count}/100")

    if illegal_count > 0:
        print(f"  ⚠️  不传递 action_masks 时会选择非法动作")
    else:
        print(f"  ✓ 即使不传递 action_masks 也正确")

    # 测试 predict with mask
    print("\n测试 2: 传递 action_masks（100次）:")
    illegal_count = 0
    action_counts = {i: 0 for i in range(9)}

    for _ in range(100):
        action, _ = model.predict(obs, deterministic=False, action_masks=mask)
        action_counts[int(action)] += 1
        if mask[action] == 0:
            illegal_count += 1

    print(f"  动作分布: {action_counts}")
    print(f"  非法动作次数: {illegal_count}/100")

    if illegal_count > 0:
        print(f"  ⚠️  MaskablePPO 传递 action_masks 后仍选择非法动作！")
    else:
        print(f"  ✓ MaskablePPO 正确使用了 action_masks")

    env.close()
    print()


def test_loaded_model():
    """测试加载的模型是否保留 masking 功能"""
    print("=" * 70)
    print("测试 3: 检查加载的模型是否使用 mask")
    print("=" * 70)

    model_path = "log/oracle_TicTacToe_ppo_masked.zip"
    if not os.path.exists(model_path):
        print(f"⚠️  模型文件不存在: {model_path}")
        print("   跳过此测试")
        return

    # 创建环境
    env = gym.make('TicTacToe-v0', opponent_type='minmax')
    env = Monitor(env)
    env = ActionMasker(env, mask_fn)

    # 加载模型
    model = MaskablePPO.load(model_path, env=env)
    print(f"✓ 模型加载成功: {model_path}")

    # 运行几局游戏
    print("\n运行 10 局测试...")
    wins, losses, draws, illegal = 0, 0, 0, 0

    for i in range(10):
        obs, _ = env.reset()
        done = False
        step_count = 0

        while not done:
            # 获取 mask
            mask = mask_fn(env)

            # 预测动作（传递 action_masks）
            action, _ = model.predict(obs, deterministic=True, action_masks=mask)

            # 检查动作是否合法
            if mask[action] == 0:
                print(f"  [局 {i+1}, 步 {step_count}] ⚠️  选择了非法动作 {action}!")
                print(f"    棋盘: {obs}")
                print(f"    Mask: {mask}")

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step_count += 1

            if done:
                if 'illegal_move' in info and info['illegal_move']:
                    illegal += 1
                    losses += 1
                    print(f"  [局 {i+1}] 非法移动导致失败")
                elif reward > 0:
                    wins += 1
                elif reward < 0:
                    losses += 1
                else:
                    draws += 1

    print(f"\n结果 (10局):")
    print(f"  胜: {wins}")
    print(f"  负: {losses}")
    print(f"  平: {draws}")
    print(f"  非法移动: {illegal}")

    env.close()
    print()


def test_policy_snapshot():
    """测试 PolicySnapshot 是否支持 action_masks"""
    print("=" * 70)
    print("测试 4: 检查 PolicySnapshot 是否支持 action_masks")
    print("=" * 70)

    # 创建环境
    env = gym.make('TicTacToe-v0', opponent_type='random')
    env = Monitor(env)
    env = ActionMasker(env, mask_fn)

    # 创建一个新的模型
    model = MaskablePPO('MlpPolicy', env, verbose=0)

    # 创建 PolicySnapshot
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    snapshot = PolicySnapshot(model.policy, device=device)
    print(f"✓ PolicySnapshot 创建成功")

    # 手动设置一个已占用的棋盘
    obs, _ = env.reset()
    if hasattr(env, 'unwrapped'):
        env.unwrapped.board[0] = 1   # 位置 0 被 X 占据
        env.unwrapped.board[4] = -1  # 位置 4 被 O 占据
        obs = env.unwrapped.board.copy()
        print(f"测试棋盘: {obs}")

    # 测试 mask
    mask = (obs == 0).astype(np.int8)
    print(f"Mask: {mask}")
    print(f"合法动作: {np.where(mask == 1)[0].tolist()}")
    print(f"非法动作: {np.where(mask == 0)[0].tolist()}")

    # 测试 predict 不带 mask
    print("\n不带 mask 的预测（100次）:")
    illegal_count = 0
    for _ in range(100):
        action, _ = snapshot.predict(obs, deterministic=False)
        if mask[action] == 0:
            illegal_count += 1
    print(f"  非法动作次数: {illegal_count}/100")

    # 测试 predict 带 mask
    print("\n带 mask 的预测（100次）:")
    illegal_count = 0
    for _ in range(100):
        action, _ = snapshot.predict(obs, deterministic=False, action_masks=mask)
        if mask[action] == 0:
            illegal_count += 1
    print(f"  非法动作次数: {illegal_count}/100")

    if illegal_count > 0:
        print(f"⚠️  PolicySnapshot 的 action_masks 未生效！")
    else:
        print(f"✓ PolicySnapshot 正确使用了 action_masks")

    env.close()
    print()


def print_prediction():
    model_path = "log/oracle_TicTacToe_ppo_masked.zip"
    if not os.path.exists(model_path):
        print(f"⚠️  模型文件不存在: {model_path}")
        print("   跳过此测试")
        return

    # 创建环境
    env = gym.make('TicTacToe-v0', opponent_type='minmax')
    env = Monitor(env)
    env = ActionMasker(env, mask_fn)

    # 加载模型
    model = MaskablePPO.load(model_path, env=env)
    print(f"✓ 模型加载成功: {model_path}")
    print('env\n', env)

    for i in range(10):
        obs, _ = env.reset()
        done = False
        step_count = 0

        while not done:
            # 获取 mask
            mask = mask_fn(env)

            # 预测动作
            action, _ = model.predict(obs, deterministic=True)

            # 检查动作是否合法
            if mask[action] == 0:
                print(f"  [局 {i+1}, 步 {step_count}] ⚠️  选择了非法动作 {action}!")
                print(f"    棋盘: {obs}")
                print(f"    Mask: {mask}")

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step_count += 1

            # if done:
            #     if 'illegal_move' in info and info['illegal_move']:
            #         illegal += 1
            #         losses += 1
            #         print(f"  [局 {i+1}] 非法移动导致失败")
            #     elif reward > 0:
            #         wins += 1
            #     elif reward < 0:
            #         losses += 1
            #     else:
            #         draws += 1





if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("MaskablePPO Action Masking 调试脚本")
    print("=" * 70)
    print()

    test_mask_function()
    test_maskable_ppo_prediction()
    test_policy_snapshot()
    test_loaded_model()

    print("=" * 70)
    print("调试完成")
    print("=" * 70)
