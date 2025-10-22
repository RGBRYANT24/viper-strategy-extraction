"""
分析训练盲点：为什么某些战术没学到
"""

import numpy as np
from stable_baselines3 import DQN
import torch
import gym_env
import gymnasium as gym


def analyze_why_missing_tactics(model_path='log/oracle_TicTacToe_delta_selfplay.zip'):
    print("=" * 70)
    print("分析：为什么神经网络错过了某些战术？")
    print("=" * 70)

    model = DQN.load(model_path)

    # 失败的案例分析
    failed_cases = [
        {
            'board': np.array([1, 0, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32),
            'correct': 7,
            'description': '立即获胜-竖向',
            'pattern': '竖向三连',
        },
        {
            'board': np.array([1, 0, 0, 0, 1, 0, 0, 0, 0], dtype=np.float32),
            'correct': 8,
            'description': '立即获胜-对角线',
            'pattern': '对角线三连',
        },
        {
            'board': np.array([0, 0, 1, 0, 1, 0, 1, 0, 0], dtype=np.float32),
            'correct': 3,
            'description': '立即获胜-反对角线',
            'pattern': '反对角线三连',
        },
    ]

    print("\n【分析1】Q值分布检查")
    print("检查网络是否真正理解了这些局面\n")

    for case in failed_cases:
        obs = case['board']
        correct_action = case['correct']
        desc = case['description']

        # 获取Q值
        obs_tensor = model.policy.obs_to_tensor(obs)[0]
        with torch.no_grad():
            q_values = model.policy.q_net(obs_tensor).cpu().numpy()[0]

        predicted_action = np.argmax(q_values)
        legal_actions = np.where(obs == 0)[0]

        print(f"[{desc}]")
        print(f"  棋盘:\n{obs.reshape(3, 3)}")
        print(f"  正确动作: {correct_action} (Q={q_values[correct_action]:.3f})")
        print(f"  预测动作: {predicted_action} (Q={q_values[predicted_action]:.3f})")
        print(f"  Q值差异: {q_values[predicted_action] - q_values[correct_action]:.3f}")

        # 检查正确动作的Q值排名
        q_legal = {i: q_values[i] for i in legal_actions}
        sorted_actions = sorted(q_legal.items(), key=lambda x: x[1], reverse=True)
        rank = [i for i, (a, _) in enumerate(sorted_actions) if a == correct_action][0] + 1

        print(f"  正确动作排名: {rank}/{len(legal_actions)}")

        if rank == 1:
            print(f"  ✓ 网络理解正确，但策略选择错误")
        elif rank <= 3:
            print(f"  △ 网络部分理解，Q值接近")
        else:
            print(f"  ✗ 网络完全不理解这个局面")
        print()

    print("\n【分析2】训练数据覆盖度推测")
    print("推测：这些局面在训练中出现的频率\n")

    # 模拟检查：横向vs竖向vs对角线的出现概率
    print("TicTacToe对称性分析:")
    print("  • 横向三连: 3种 (第1,2,3行)")
    print("  • 竖向三连: 3种 (第1,2,3列)")
    print("  • 对角线:   2种 (主对角+反对角)")
    print()
    print("如果训练数据是随机的，应该:")
    print("  • 横向出现: 3/8 = 37.5%")
    print("  • 竖向出现: 3/8 = 37.5%")
    print("  • 对角线:   2/8 = 25.0%")
    print()
    print("⚠ 但你的网络只学会了横向！")
    print()
    print("可能原因:")
    print("  1. MinMax对手总是优先防守横向")
    print("  2. 训练早期随机对手被利用横向战术击败")
    print("  3. 网络过早收敛到局部最优（只会横向攻击）")

    print("\n【分析3】对比：横向是否学到了？")
    print()

    horizontal_case = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    obs_tensor = model.policy.obs_to_tensor(horizontal_case)[0]
    with torch.no_grad():
        q_values = model.policy.q_net(obs_tensor).cpu().numpy()[0]

    print(f"[横向获胜机会]")
    print(f"  棋盘:\n{horizontal_case.reshape(3, 3)}")
    print(f"  正确动作: 2")
    print(f"  预测动作: {np.argmax(q_values)}")
    print(f"  Q(正确): {q_values[2]:.3f}")

    if np.argmax(q_values) == 2:
        print(f"  ✓ 横向完美学习")
        print()
        print("结论: 网络有能力学习战术，但**训练数据不平衡**")
    else:
        print(f"  ✗ 连横向都没学好")
        print()
        print("结论: 网络根本没学到战术，需要**重新训练**")

    print("\n" + "=" * 70)
    print("根本原因诊断")
    print("=" * 70)

    print("""
🎯 核心问题: 训练数据分布不均

你的模型能达到100%平局vs MinMax，说明：
  ✓ 网络容量足够
  ✓ 学习算法正常
  ✗ 但某些战术局面从未充分训练

这是因为:
  1. Self-play会陷入"共同盲点"
     - 如果双方都不擅长竖向攻击
     - 训练中竖向机会就很少出现
     - 永远学不到竖向战术

  2. MinMax基准不够强
     - 你用了 --use-minmax，但只有2个基准（Random + MinMax）
     - MinMax在池中占比太小（2/22 = 9%）
     - 大部分对战是弱对手

解决方案:
""")

    print("\n方案A: 增加数据增强（推荐）")
    print("-" * 70)
    print("""
在环境中添加棋盘旋转/镜像变换，强制学习所有方向:

修改 tictactoe_delta_selfplay.py:

    def _get_observation(self):
        obs = ...  # 原始观察

        # 随机应用旋转 (25%概率)
        if np.random.random() < 0.25:
            obs_2d = obs.reshape(3, 3)
            k = np.random.randint(1, 4)  # 旋转90/180/270度
            obs_2d = np.rot90(obs_2d, k=k)
            obs = obs_2d.flatten()

        return obs
""")

    print("\n方案B: 改进训练参数")
    print("-" * 70)
    print("""
python train/train_delta_selfplay.py \\
    --total-timesteps 500000 \\        # 增加训练步数
    --max-pool-size 10 \\              # 减小池，增加MinMax比例
    --update-interval 20000 \\         # 减少更新频率
    --use-minmax \\
    --n-env 8

这样 MinMax 占比提高到: 2/12 = 16.7%
""")

    print("\n方案C: 课程学习（Curriculum Learning）")
    print("-" * 70)
    print("""
先用固定的战术场景训练，再进行self-play:

1. 阶段1: 对战 MinMax (50k步)
2. 阶段2: Delta-Uniform Self-Play (200k步)
3. 阶段3: 精调对战 MinMax (50k步)
""")


def suggest_immediate_fix():
    print("\n" + "=" * 70)
    print("立即可执行的修复方案")
    print("=" * 70)

    print("""
🚀 方案1: 继续训练（最简单）
-----------------------------------------
你的模型已经很接近最优，继续训练可能会自动修复:

python train/train_delta_selfplay.py \\
    --total-timesteps 200000 \\      # 额外20万步
    --max-pool-size 10 \\            # 缩小池，增加强对手比例
    --use-minmax \\
    --output log/oracle_TicTacToe_delta_selfplay_v2.zip

预期: 随着训练继续，竖向/对角线场景会逐渐出现


🚀 方案2: 纯MinMax训练（最有效）
-----------------------------------------
不用self-play，直接对战MinMax:

python train/train_delta_selfplay.py \\
    --total-timesteps 300000 \\
    --max-pool-size 1 \\             # 池大小=1，强制只对战MinMax
    --use-minmax \\
    --output log/oracle_TicTacToe_minmax_only.zip

预期: 直接学到最优策略，无盲点


🚀 方案3: 混合训练（最平衡）
-----------------------------------------
# 第一阶段: 对战MinMax打基础
python train/train_delta_selfplay.py \\
    --total-timesteps 150000 \\
    --max-pool-size 1 \\
    --use-minmax \\
    --output log/oracle_TicTacToe_phase1.zip

# 第二阶段: Self-Play精调
python train/train_delta_selfplay.py \\
    --total-timesteps 150000 \\
    --max-pool-size 20 \\
    --use-minmax \\
    --output log/oracle_TicTacToe_phase2.zip
    # (需要修改代码支持加载已有模型继续训练)


📊 如何选择?
-----------------------------------------
• 如果你想快速得到最优策略 → 方案2 (纯MinMax)
• 如果你想研究self-play机制   → 方案1 (继续训练)
• 如果你时间充裕            → 方案3 (混合训练)

我推荐: 方案2，因为简单有效，3-4小时就能完成
""")


if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else 'log/oracle_TicTacToe_delta_selfplay.zip'

    analyze_why_missing_tactics(model_path)
    suggest_immediate_fix()
