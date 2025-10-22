"""
检查所有位置的Q值，包括非法位置
验证训练时是否正确处理了非法动作
"""

import numpy as np
from stable_baselines3 import DQN
import torch
import gym_env


def check_illegal_position_q_values(model_path='log/oracle_TicTacToe_delta_selfplay.zip'):
    """检查非法位置的Q值"""

    print("=" * 70)
    print("检查非法位置的Q值")
    print("=" * 70)

    model = DQN.load(model_path)

    # 测试场景：已有两个棋子
    obs = np.array([1, 0, -1, 0, 0, 0, 0, 0, 0], dtype=np.float32)

    print(f"\n棋盘状态:")
    print(obs.reshape(3, 3))
    print(f"\n合法位置: {np.where(obs == 0)[0]}")
    print(f"非法位置: {np.where(obs != 0)[0]}")

    # 获取所有位置的Q值
    obs_tensor = model.policy.obs_to_tensor(obs)[0]
    with torch.no_grad():
        q_values = model.policy.q_net(obs_tensor).cpu().numpy()[0]

    print(f"\n所有位置的Q值:")
    for i in range(9):
        is_legal = (obs[i] == 0)
        status = "✓ 合法" if is_legal else "✗ 非法"
        print(f"  位置 {i}: Q={q_values[i]:7.3f}  {status}")

    # 分析
    legal_q = q_values[obs == 0]
    illegal_q = q_values[obs != 0]

    print(f"\n统计:")
    print(f"  合法位置Q值: mean={legal_q.mean():.3f}, std={legal_q.std():.3f}")
    print(f"  非法位置Q值: mean={illegal_q.mean():.3f}, std={illegal_q.std():.3f}")

    # 关键检查
    print(f"\n关键检查:")

    # 1. 非法位置的Q值是否很低？
    if illegal_q.mean() < legal_q.mean() - 1.0:
        print(f"  ✓ 非法位置Q值明显低于合法位置")
        print(f"    → 模型学到了不选非法动作")
    else:
        print(f"  ⚠ 非法位置Q值不够低")
        print(f"    → 可能在训练时没有正确mask")

    # 2. argmax 是否会选到非法位置？
    best_action = np.argmax(q_values)
    if best_action in np.where(obs == 0)[0]:
        print(f"  ✓ argmax选择了合法动作 ({best_action})")
    else:
        print(f"  ✗ argmax选择了非法动作 ({best_action})")
        print(f"    → 需要mask！")

    return q_values, obs


def check_training_target_computation():
    """
    理论分析：训练时目标Q值的计算
    """
    print("\n" + "=" * 70)
    print("训练时目标Q值计算分析")
    print("=" * 70)

    print("""
标准DQN训练（Stable-Baselines3）:

1. 采样转移: (s, a, r, s', done)

2. 计算目标Q值:
   target_q = r + γ * max_{a'} Q_target(s', a')
                     ^^^^^^^^
                     这里会选所有9个动作中的最大值
                     包括非法动作！

3. 如果 s' 中某些位置已占用:
   - 这些位置的动作是非法的
   - 但 max 操作仍会考虑它们的Q值
   - 如果非法动作的Q值高，会被选中

4. 结果:
   - 如果非法动作Q值 = -10 (初期探索到的)
     → target_q 被拉低
     → 当前Q值也会被训练成负数

   - 如果非法动作Q值 = 高值 (错误学习)
     → target_q 被拉高
     → 学到错误的策略

结论: 需要在计算 max Q(s', a') 时mask掉非法动作！
""")


def propose_solution():
    """提出解决方案"""
    print("\n" + "=" * 70)
    print("解决方案")
    print("=" * 70)

    print("""
方案1: 使用 Maskable PPO（推荐）⭐
-------------------------------------
stable-baselines3-contrib 提供了支持action masking的算法

安装:
  pip install sb3-contrib

使用:
  from sb3_contrib import MaskablePPO
  from sb3_contrib.common.envs import InvalidActionEnvDiscrete

  model = MaskablePPO("MlpPolicy", env, ...)
  model.learn(total_timesteps=300000)

优点:
  • 原生支持action masking
  • 训练和推理都会mask
  • 不需要修改环境
  • 稳定可靠


方案2: 自定义DQN (复杂)
-------------------------------------
继承 stable_baselines3.DQN，重写 train() 方法

需要修改:
  def train(self, gradient_steps):
      # ... 原有代码 ...

      # 修改目标Q值计算
      with torch.no_grad():
          next_q_values = self.q_net_target(next_observations)

          # ⭐ 添加mask
          action_masks = self._get_action_masks(next_observations)
          next_q_values[action_masks == 0] = -float('inf')

          next_q_values, _ = next_q_values.max(dim=1)

      # ... 剩余代码 ...

缺点:
  • 需要大量代码
  • 维护困难
  • 容易出bug


方案3: 预处理经验（折衷）
-------------------------------------
在存入replay buffer之前，确保所有经验都是合法的

修改环境的step():
  if not is_valid_action(action):
      # 不是返回-10，而是选择最近的合法动作
      action = find_closest_legal_action(action)
      # 然后正常执行

优点:
  • 简单
  • 不需要修改训练代码

缺点:
  • 治标不治本
  • Q值仍可能不准确


我的推荐: 方案1（MaskablePPO）
-------------------------------------
最简单、最可靠的方案
""")

    print("\n立即可执行:")
    print("  pip install sb3-contrib")
    print("  # 然后修改训练脚本使用 MaskablePPO")


if __name__ == "__main__":
    import sys

    model_path = sys.argv[1] if len(sys.argv) > 1 else 'log/oracle_TicTacToe_delta_selfplay.zip'

    check_illegal_position_q_values(model_path)
    check_training_target_computation()
    propose_solution()
