"""
诊断 Delta-Uniform Self-Play 训练问题
帮助定位为什么训练效果不好
"""

import argparse
import numpy as np
from stable_baselines3 import DQN
import gymnasium as gym
from gym_env.tictactoe_delta_selfplay import TicTacToeDeltaSelfPlayEnv
from gym_env.policies import RandomPlayerPolicy, MinMaxPlayerPolicy
from gymnasium import spaces


def test_model_behavior(model_path, num_games=10):
    """测试模型的具体行为"""
    print("=" * 70)
    print("诊断 1: 模型行为分析")
    print("=" * 70)

    model = DQN.load(model_path)

    # 测试场景1: 空棋盘
    print("\n[场景1] 空棋盘 - 应该选择中心(4)或角落(0,2,6,8)")
    obs = np.zeros(9, dtype=np.float32)
    action, _ = model.predict(obs, deterministic=True)
    print(f"  棋盘: {obs.reshape(3, 3)}")
    print(f"  预测动作: {action}")
    print(f"  评价: {'✓ 合理' if action in [0, 2, 4, 6, 8] else '✗ 不佳'}")

    # 测试场景2: 简单获胜机会
    print("\n[场景2] 两子连线 - 应该选择2获胜")
    obs = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    action, _ = model.predict(obs, deterministic=True)
    print(f"  棋盘:\n{obs.reshape(3, 3)}")
    print(f"  预测动作: {action}")
    print(f"  评价: {'✓ 正确' if action == 2 else '✗ 错误'}")

    # 测试场景3: 防守
    print("\n[场景3] 对手两子连线 - 应该选择2防守")
    obs = np.array([-1, -1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    action, _ = model.predict(obs, deterministic=True)
    print(f"  棋盘:\n{obs.reshape(3, 3)}")
    print(f"  预测动作: {action}")
    print(f"  评价: {'✓ 正确' if action == 2 else '✗ 错误'}")

    # 测试场景4: 中局复杂局面
    print("\n[场景4] 中局复杂局面")
    obs = np.array([1, 0, -1, 0, 1, 0, 0, 0, -1], dtype=np.float32)
    action, _ = model.predict(obs, deterministic=True)
    legal_actions = np.where(obs == 0)[0]
    print(f"  棋盘:\n{obs.reshape(3, 3)}")
    print(f"  合法动作: {legal_actions}")
    print(f"  预测动作: {action}")
    print(f"  评价: {'✓ 合法' if action in legal_actions else '✗ 非法'}")

    # 测试Q值分布
    print("\n[场景5] Q值分析 (空棋盘)")
    obs = np.zeros(9, dtype=np.float32)
    obs_tensor = model.policy.obs_to_tensor(obs)[0]
    with model.policy.policy.q_net.eval():
        import torch
        q_values = model.policy.q_net(obs_tensor).detach().cpu().numpy()[0]

    print(f"  Q值: {q_values}")
    print(f"  最大Q值位置: {np.argmax(q_values)}")
    print(f"  Q值范围: [{q_values.min():.3f}, {q_values.max():.3f}]")
    print(f"  Q值标准差: {q_values.std():.3f}")

    # 检查Q值是否都很接近（说明没学到东西）
    if q_values.std() < 0.1:
        print(f"  ⚠ Q值标准差很小 ({q_values.std():.3f}) - 可能没有充分学习")

    return model


def test_vs_random(model_path, num_games=50):
    """测试对战随机对手"""
    print("\n" + "=" * 70)
    print("诊断 2: 对战随机对手")
    print("=" * 70)

    model = DQN.load(model_path)
    env = gym.make('TicTacToe-v0', opponent_type='random')

    wins = 0
    losses = 0
    draws = 0
    illegal_moves = 0

    for i in range(num_games):
        obs, _ = env.reset()
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if done:
                if 'illegal_move' in info and info['illegal_move']:
                    illegal_moves += 1
                    losses += 1
                elif reward > 0:
                    wins += 1
                elif reward < 0:
                    losses += 1
                else:
                    draws += 1

    env.close()

    print(f"\n结果 ({num_games} 局 vs Random):")
    print(f"  胜: {wins} ({wins/num_games*100:.1f}%)")
    print(f"  负: {losses} ({losses/num_games*100:.1f}%)")
    print(f"  平: {draws} ({draws/num_games*100:.1f}%)")
    print(f"  非法移动: {illegal_moves}")

    if illegal_moves > 0:
        print(f"\n  ⚠ 有非法移动 - 模型有严重问题")
        return False
    elif wins < 30:
        print(f"\n  ⚠ 胜率太低 - 连随机对手都打不过")
        return False
    else:
        print(f"\n  ✓ 能战胜随机对手 - 基本策略已学到")
        return True


def test_vs_minmax(model_path, num_games=50):
    """测试对战MinMax"""
    print("\n" + "=" * 70)
    print("诊断 3: 对战 MinMax (详细)")
    print("=" * 70)

    model = DQN.load(model_path)
    env = gym.make('TicTacToe-v0', opponent_type='minmax')

    wins = 0
    losses = 0
    draws = 0
    illegal_moves = 0
    first_moves = {}

    for i in range(num_games):
        obs, _ = env.reset()
        done = False
        first_move = None

        while not done:
            action, _ = model.predict(obs, deterministic=True)

            if first_move is None:
                first_move = action
                first_moves[action] = first_moves.get(action, 0) + 1

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if done:
                if 'illegal_move' in info and info['illegal_move']:
                    illegal_moves += 1
                    losses += 1
                elif reward > 0:
                    wins += 1
                elif reward < 0:
                    losses += 1
                else:
                    draws += 1

    env.close()

    print(f"\n结果 ({num_games} 局 vs MinMax):")
    print(f"  胜: {wins} ({wins/num_games*100:.1f}%)")
    print(f"  负: {losses} ({losses/num_games*100:.1f}%)")
    print(f"  平: {draws} ({draws/num_games*100:.1f}%)")
    print(f"  非法移动: {illegal_moves}")

    print(f"\n首步分布:")
    for pos, count in sorted(first_moves.items(), key=lambda x: x[1], reverse=True):
        print(f"  位置 {pos}: {count} 次 ({count/num_games*100:.1f}%)")

    # 分析首步选择
    if 4 in first_moves and first_moves[4] > num_games * 0.8:
        print(f"\n  ✓ 首步主要选择中心(4) - 策略合理")
    elif all(pos in [0, 2, 6, 8] for pos in first_moves.keys()):
        print(f"\n  △ 首步选择角落 - 次优但可接受")
    else:
        print(f"\n  ⚠ 首步选择分散 - 策略不稳定")

    return draws >= 40


def test_exploration_rate(model_path):
    """检查探索率"""
    print("\n" + "=" * 70)
    print("诊断 4: 训练参数检查")
    print("=" * 70)

    model = DQN.load(model_path)

    print(f"\n探索参数:")
    print(f"  当前探索率 (epsilon): {model.exploration_rate:.4f}")
    print(f"  最终探索率: {model.exploration_final_eps:.4f}")
    print(f"  探索衰减比例: {model.exploration_fraction}")

    print(f"\n学习参数:")
    print(f"  学习率: {model.learning_rate}")
    print(f"  批大小: {model.batch_size}")
    print(f"  缓冲区大小: {model.buffer_size}")
    print(f"  已训练步数: {model.num_timesteps}")

    print(f"\n网络结构:")
    print(f"  {model.policy.net_arch}")

    # 检查经验回放池
    if hasattr(model, 'replay_buffer'):
        buffer_size = model.replay_buffer.size()
        print(f"\n经验回放池:")
        print(f"  当前大小: {buffer_size}")
        print(f"  容量: {model.buffer_size}")

        if buffer_size < 10000:
            print(f"  ⚠ 经验池太小 - 可能没有充分学习")


def visualize_game(model_path):
    """可视化一局游戏"""
    print("\n" + "=" * 70)
    print("诊断 5: 可视化对战 (vs MinMax)")
    print("=" * 70)

    model = DQN.load(model_path)
    env = gym.make('TicTacToe-v0', opponent_type='minmax')

    obs, _ = env.reset(seed=42)
    done = False
    step = 0

    print(f"\n初始状态:")
    print(f"{obs.reshape(3, 3)}\n")

    while not done and step < 10:
        action, _ = model.predict(obs, deterministic=True)

        # 获取Q值
        obs_tensor = model.policy.obs_to_tensor(obs)[0]
        import torch
        with torch.no_grad():
            q_values = model.policy.q_net(obs_tensor).cpu().numpy()[0]

        print(f"步骤 {step + 1}:")
        print(f"  Q值: {q_values}")
        print(f"  选择动作: {action} (Q={q_values[action]:.3f})")

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        print(f"  棋盘状态:")
        print(f"{obs.reshape(3, 3)}")

        if done:
            print(f"\n游戏结束:")
            print(f"  奖励: {reward}")
            print(f"  信息: {info}")

        print()
        step += 1

    env.close()


def main():
    parser = argparse.ArgumentParser(description='诊断 Delta-Uniform Self-Play 训练问题')
    parser.add_argument('--model', type=str,
                        default='log/oracle_TicTacToe_delta_selfplay.zip',
                        help='模型路径')
    parser.add_argument('--num-games', type=int, default=50,
                        help='测试局数')
    parser.add_argument('--visualize', action='store_true',
                        help='可视化一局游戏')
    args = parser.parse_args()

    print()
    print("=" * 70)
    print("Delta-Uniform Self-Play 训练诊断工具")
    print("=" * 70)
    print(f"\n模型: {args.model}")
    print()

    # 运行诊断
    try:
        # 1. 模型行为
        test_model_behavior(args.model)

        # 2. vs Random
        can_beat_random = test_vs_random(args.model, args.num_games)

        # 3. vs MinMax
        can_draw_minmax = test_vs_minmax(args.model, args.num_games)

        # 4. 参数检查
        test_exploration_rate(args.model)

        # 5. 可视化
        if args.visualize:
            visualize_game(args.model)

        # 总结
        print("\n" + "=" * 70)
        print("诊断总结")
        print("=" * 70)

        if not can_beat_random:
            print("\n⚠ 核心问题: 连随机对手都打不过")
            print("\n可能原因:")
            print("  1. 训练步数不足")
            print("  2. 学习率设置不当")
            print("  3. 探索策略问题")
            print("\n建议:")
            print("  1. 增加训练步数到 500k")
            print("  2. 检查学习率 (尝试 1e-3 或 1e-4)")
            print("  3. 延长探索时间 (exploration_fraction=0.7)")

        elif not can_draw_minmax:
            print("\n⚠ 问题: 能战胜Random但无法平局MinMax")
            print("\n可能原因:")
            print("  1. 对手池中没有强对手")
            print("  2. 池大小太小，多样性不足")
            print("  3. 没有训练后手")
            print("\n建议:")
            print("  1. 使用 --use-minmax 添加MinMax基准")
            print("  2. 增加池大小到 30-50")
            print("  3. 确保 play_as_o_prob=0.5")

        else:
            print("\n✓ 模型训练良好！")
            print(f"\n性能:")
            print(f"  vs Random: 能获胜")
            print(f"  vs MinMax: 高平局率")

        print()
        print("=" * 70)

    except Exception as e:
        print(f"\n✗ 诊断过程出错: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
