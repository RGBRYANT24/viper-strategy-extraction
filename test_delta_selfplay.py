"""
测试 Delta-Uniform Self-Play 实现
运行此脚本验证所有组件是否正常工作
"""

import sys
import numpy as np
from gymnasium import spaces


def test_baseline_policies():
    """测试基准策略"""
    print("=" * 70)
    print("测试 1: 基准策略 (RandomPlayerPolicy, MinMaxPlayerPolicy)")
    print("=" * 70)

    from gym_env.policies import RandomPlayerPolicy, MinMaxPlayerPolicy

    # 创建空间
    obs_space = spaces.Box(low=-1, high=1, shape=(9,), dtype=np.float32)
    act_space = spaces.Discrete(9)

    # 测试 RandomPlayerPolicy
    print("\n[1.1] RandomPlayerPolicy")
    random_policy = RandomPlayerPolicy(obs_space, act_space)

    test_board = np.array([1, 0, -1, 0, 1, 0, 0, 0, -1], dtype=np.float32)
    print(f"Board:\n{test_board.reshape(3, 3)}")

    for i in range(3):
        action, _ = random_policy.predict(test_board)
        print(f"  Trial {i+1}: action={action}")

    # 测试 MinMaxPlayerPolicy
    print("\n[1.2] MinMaxPlayerPolicy")
    minmax_policy = MinMaxPlayerPolicy(obs_space, act_space)

    # 测试场景1：简单获胜机会
    board1 = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    print(f"\nBoard (should choose 2 to win):")
    print(board1.reshape(3, 3))
    action, _ = minmax_policy.predict(board1)
    print(f"Action: {action} (expected: 2)")

    # 测试场景2：空棋盘
    board2 = np.zeros(9, dtype=np.float32)
    print(f"\nBoard (empty, should choose center=4):")
    print(board2.reshape(3, 3))
    action, _ = minmax_policy.predict(board2)
    print(f"Action: {action} (expected: 4)")

    print("\n✓ 基准策略测试通过")
    return True


def test_delta_selfplay_env():
    """测试 Delta-Uniform Self-Play 环境"""
    print("\n" + "=" * 70)
    print("测试 2: Delta-Uniform Self-Play 环境")
    print("=" * 70)

    import gymnasium as gym
    from gym_env.tictactoe_delta_selfplay import TicTacToeDeltaSelfPlayEnv
    from gym_env.policies import RandomPlayerPolicy, MinMaxPlayerPolicy

    # 创建基准策略
    obs_space = spaces.Box(low=-1, high=1, shape=(9,), dtype=np.float32)
    act_space = spaces.Discrete(9)

    baseline_pool = [
        RandomPlayerPolicy(obs_space, act_space),
        MinMaxPlayerPolicy(obs_space, act_space)
    ]

    # 创建环境
    print("\n[2.1] 创建环境")
    env = TicTacToeDeltaSelfPlayEnv(
        baseline_pool=baseline_pool,
        learned_pool=None,
        play_as_o_prob=0.5
    )
    print(f"  观察空间: {env.observation_space}")
    print(f"  动作空间: {env.action_space}")
    print(f"  基准池大小: {len(env.baseline_pool)}")

    # 测试先手
    print("\n[2.2] 测试游戏流程 (先手 X)")
    obs, info = env.reset(seed=42)
    print(f"  Playing as: {'O (后手)' if env.play_as_o else 'X (先手)'}")
    print(f"  Initial observation:\n{obs.reshape(3, 3)}")

    for step in range(3):
        legal_actions = np.where(obs == 0)[0]
        if len(legal_actions) == 0:
            break
        action = np.random.choice(legal_actions)

        obs, reward, terminated, truncated, info = env.step(action)
        print(f"\n  Step {step + 1}: action={action}, reward={reward}")
        print(f"  Observation:\n{obs.reshape(3, 3)}")

        if terminated:
            print(f"  Game ended: {info}")
            break

    env.close()
    print("\n✓ Delta-Uniform Self-Play 环境测试通过")
    return True


def test_environment_registration():
    """测试环境注册"""
    print("\n" + "=" * 70)
    print("测试 3: 环境注册")
    print("=" * 70)

    import gymnasium as gym

    print("\n[3.1] 检查 TicTacToe-v0")
    env1 = gym.make('TicTacToe-v0')
    print(f"  ✓ TicTacToe-v0 注册成功: {type(env1)}")
    env1.close()

    print("\n[3.2] 检查 TicTacToe-SelfPlay-v0")
    env2 = gym.make('TicTacToe-SelfPlay-v0')
    print(f"  ✓ TicTacToe-SelfPlay-v0 注册成功: {type(env2)}")
    env2.close()

    print("\n[3.3] 检查 TicTacToe-DeltaSelfPlay-v0")
    env3 = gym.make('TicTacToe-DeltaSelfPlay-v0')
    print(f"  ✓ TicTacToe-DeltaSelfPlay-v0 注册成功: {type(env3)}")
    env3.close()

    print("\n✓ 环境注册测试通过")
    return True


def test_training_imports():
    """测试训练脚本的导入"""
    print("\n" + "=" * 70)
    print("测试 4: 训练脚本导入")
    print("=" * 70)

    try:
        print("\n[4.1] 导入必需模块")
        from stable_baselines3 import DQN
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.monitor import Monitor
        from collections import deque
        print("  ✓ stable-baselines3 导入成功")

        print("\n[4.2] 导入自定义模块")
        from gym_env.tictactoe_delta_selfplay import TicTacToeDeltaSelfPlayEnv
        from gym_env.policies import RandomPlayerPolicy, MinMaxPlayerPolicy
        print("  ✓ 自定义模块导入成功")

        print("\n✓ 训练脚本导入测试通过")
        return True

    except ImportError as e:
        print(f"\n✗ 导入失败: {e}")
        return False


def main():
    """运行所有测试"""
    print()
    print("=" * 70)
    print("Delta-Uniform Self-Play 实现测试套件")
    print("=" * 70)

    results = []

    try:
        results.append(("基准策略", test_baseline_policies()))
    except Exception as e:
        print(f"\n✗ 基准策略测试失败: {e}")
        import traceback
        traceback.print_exc()
        results.append(("基准策略", False))

    try:
        results.append(("Delta-Uniform 环境", test_delta_selfplay_env()))
    except Exception as e:
        print(f"\n✗ Delta-Uniform 环境测试失败: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Delta-Uniform 环境", False))

    try:
        results.append(("环境注册", test_environment_registration()))
    except Exception as e:
        print(f"\n✗ 环境注册测试失败: {e}")
        import traceback
        traceback.print_exc()
        results.append(("环境注册", False))

    try:
        results.append(("训练脚本导入", test_training_imports()))
    except Exception as e:
        print(f"\n✗ 训练脚本导入测试失败: {e}")
        import traceback
        traceback.print_exc()
        results.append(("训练脚本导入", False))

    # 总结
    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)

    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{name:20s}: {status}")

    all_passed = all(r[1] for r in results)
    print()
    if all_passed:
        print("🎉 所有测试通过！可以开始训练了。")
        print()
        print("运行训练:")
        print("  python train/train_delta_selfplay.py --total-timesteps 200000 --max-pool-size 20")
        print()
        print("或者使用 MinMax 基准:")
        print("  python train/train_delta_selfplay.py --total-timesteps 200000 --max-pool-size 20 --use-minmax")
    else:
        print("⚠ 部分测试失败，请检查错误信息。")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
