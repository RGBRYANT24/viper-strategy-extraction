"""
快速诊断：3个关键测试找出问题
"""

from stable_baselines3 import DQN
import numpy as np
import gymnasium as gym

def quick_diagnose(model_path='log/oracle_TicTacToe_delta_selfplay.zip'):
    print("快速诊断 Delta-Uniform Self-Play 模型")
    print("=" * 70)

    try:
        model = DQN.load(model_path)
        print(f"✓ 模型加载成功: {model_path}\n")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        return

    # 测试1: 空棋盘应该选中心或角落
    print("【测试1】空棋盘策略")
    obs = np.zeros(9, dtype=np.float32)
    action, _ = model.predict(obs, deterministic=True)
    print(f"  棋盘: 空")
    print(f"  选择: 位置 {action}")
    print(f"  期望: 中心(4) 或 角落(0,2,6,8)")
    print(f"  结果: {'✓ 合理' if action in [0,2,4,6,8] else '✗ 不合理'}\n")

    # 测试2: 明显的获胜机会
    print("【测试2】获胜机会识别")
    obs = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    action, _ = model.predict(obs, deterministic=True)
    print(f"  棋盘: X X .")
    print(f"        . . .")
    print(f"        . . .")
    print(f"  选择: 位置 {action}")
    print(f"  期望: 位置 2 (获胜)")
    print(f"  结果: {'✓ 正确' if action == 2 else '✗ 错误'}\n")

    # 测试3: 对手威胁防守
    print("【测试3】防守意识")
    obs = np.array([-1, -1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    action, _ = model.predict(obs, deterministic=True)
    print(f"  棋盘: O O .")
    print(f"        . . .")
    print(f"        . . .")
    print(f"  选择: 位置 {action}")
    print(f"  期望: 位置 2 (防守)")
    print(f"  结果: {'✓ 正确' if action == 2 else '✗ 错误'}\n")

    # 测试4: Q值检查
    print("【测试4】Q值分布 (空棋盘)")
    obs = np.zeros(9, dtype=np.float32)
    obs_tensor = model.policy.obs_to_tensor(obs)[0]
    import torch
    with torch.no_grad():
        q_values = model.policy.q_net(obs_tensor).cpu().numpy()[0]

    print(f"  Q值: {np.round(q_values, 3)}")
    print(f"  最大: {q_values.max():.3f}, 最小: {q_values.min():.3f}")
    print(f"  标准差: {q_values.std():.3f}")

    if q_values.std() < 0.1:
        print(f"  ⚠ 警告: Q值几乎相同 (std={q_values.std():.4f})")
        print(f"      → 模型可能没有学到任何东西")
    elif q_values.max() - q_values.min() > 5:
        print(f"  ✓ Q值有明显差异，模型有区分能力")
    else:
        print(f"  △ Q值有一定差异，但不够显著")

    print("\n" + "=" * 70)
    print("快速建议:")
    print("=" * 70)

    # 对Random测试
    print("\n【额外测试】对战 Random (10局)")
    env = gym.make('TicTacToe-v0', opponent_type='random')
    wins, losses, draws = 0, 0, 0

    for _ in range(10):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if done:
                if reward > 0: wins += 1
                elif reward < 0: losses += 1
                else: draws += 1

    print(f"  胜:{wins} 平:{draws} 负:{losses}")

    if wins < 5:
        print("\n❌ 核心问题: 连Random都打不过")
        print("\n可能原因:")
        print("  1. 对手池全是弱对手 (只有Random)")
        print("  2. 训练步数严重不足")
        print("  3. 探索率衰减太快")
        print("\n解决方案:")
        print("  方案A (推荐): 重新训练，添加MinMax基准")
        print("    python train/train_delta_selfplay.py \\")
        print("      --total-timesteps 300000 \\")
        print("      --max-pool-size 20 \\")
        print("      --use-minmax \\")
        print("      --n-env 8")
        print("\n  方案B: 使用更长的探索时间")
        print("    python train/train_delta_selfplay.py \\")
        print("      --total-timesteps 500000 \\")
        print("      --max-pool-size 30")

    elif draws < 3:
        print("\n⚠ 问题: 能赢Random但无法平MinMax")
        print("\n原因: 对手池缺乏强对手")
        print("\n解决方案:")
        print("  python train/train_delta_selfplay.py \\")
        print("    --total-timesteps 300000 \\")
        print("    --max-pool-size 30 \\")
        print("    --use-minmax")

    else:
        print("\n✓ 模型基本正常，继续训练可能会更好")

    env.close()
    print()

if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else 'log/oracle_TicTacToe_delta_selfplay.zip'
    quick_diagnose(model_path)
