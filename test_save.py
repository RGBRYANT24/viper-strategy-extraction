"""
测试模型保存功能
"""
import os
from stable_baselines3 import DQN
from gym_env import make_env
import argparse

# 模拟 args
class Args:
    env_name = "TicTacToe-v0"
    n_env = 1
    ep_horizon = 150
    rand_ball_start = False
    render = False

args = Args()

print("=" * 60)
print("测试 TicTacToe 环境和 DQN 保存")
print("=" * 60)

# 1. 创建环境
print("\n1. 创建环境...")
env = make_env(args)
print(f"   ✓ 环境创建成功: {env}")

# 2. 创建简单的 DQN 模型
print("\n2. 创建 DQN 模型...")
model = DQN(
    'MlpPolicy',
    env,
    learning_rate=1e-3,
    buffer_size=1000,
    batch_size=32,
    verbose=1
)
print(f"   ✓ 模型创建成功")

# 3. 训练一点点
print("\n3. 训练 1000 步...")
model.learn(total_timesteps=1000, log_interval=1000)
print(f"   ✓ 训练完成")

# 4. 测试保存
print("\n4. 测试保存...")
save_paths = [
    "./log/test_oracle_TicTacToe-v0",
    "./oracle_test",
]

for path in save_paths:
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        print(f"\n   尝试保存到: {path}")
        model.save(path)

        # 检查文件是否存在
        zip_path = f"{path}.zip"
        if os.path.exists(zip_path):
            size = os.path.getsize(zip_path)
            print(f"   ✓ 保存成功! 文件: {zip_path} ({size} bytes)")
        else:
            print(f"   ✗ 保存失败! 文件不存在: {zip_path}")

    except Exception as e:
        print(f"   ✗ 保存出错: {e}")

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)
