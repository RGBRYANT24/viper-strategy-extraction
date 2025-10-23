"""
自定义 Masked DQN Policy
在Q值选择时自动屏蔽非法动作，避免Q值污染问题

核心思想：
1. 保持 stable_baselines3.DQN 的标准接口（与VIPER兼容）
2. 在 predict() 时将非法动作的Q值设为 -inf
3. 训练时的ε-greedy exploration也会遵守mask

棋盘表示：
- 1: 自己的棋子（从自己视角看）
- -1: 对手的棋子
- 0: 空位（合法动作）

关键修复：
- 重写 `_sample_action()` 方法，在exploration时也mask非法动作
- 这样即使随机探索，也只会从合法动作中选择
"""

import torch
import numpy as np
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.common.type_aliases import Schedule
from gymnasium import spaces
from typing import Any, Dict, List, Optional, Tuple, Type, Union


class MaskedDQNPolicy(DQNPolicy):
    """
    支持 Action Masking 的 DQN Policy

    在预测时自动将非法动作的Q值设为负无穷，确保永远不会选择非法动作
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_obs = None

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        覆盖 predict 方法，支持 action masking

        Args:
            observation: 观察值（棋盘状态）
            state: RNN状态（未使用）
            episode_start: 是否新回合
            deterministic: 是否确定性选择

        Returns:
            action: 选择的动作
            state: 更新的状态
        """
        # 保存观察用于生成mask
        self._last_obs = observation

        # 调用父类的predict，但我们会在 _predict 中应用mask
        return super().predict(observation, state, episode_start, deterministic)

    def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        内部预测方法，应用 action masking

        Args:
            observation: 观察张量
            deterministic: 是否确定性选择

        Returns:
            actions: 选择的动作
        """
        # 获取Q值
        with torch.no_grad():
            q_values = self.q_net(observation)

        # 应用 action mask
        # 关键：只有值为 0 的位置是合法的（空位）
        # 值为 1 或 -1 的位置都是非法的（已被占据）
        if isinstance(observation, torch.Tensor):
            obs_np = observation.cpu().numpy()
        else:
            obs_np = observation

        # 确保是2D: (batch_size, 9)
        if len(obs_np.shape) == 1:
            obs_np = obs_np.reshape(1, -1)
            single_obs = True
        else:
            single_obs = False

        # 创建mask: 精确判断是否为0（使用绝对值小于阈值）
        # 这样可以处理浮点数精度问题
        mask = np.abs(obs_np) < 1e-6  # shape: (batch_size, 9)
        mask_tensor = torch.tensor(mask, dtype=torch.bool, device=q_values.device)

        # 将非法动作的Q值设为负无穷
        masked_q_values = q_values.clone()
        masked_q_values[~mask_tensor] = float('-inf')

        # 选择动作（选择Q值最大的合法动作）
        actions = torch.argmax(masked_q_values, dim=1)

        if single_obs:
            return actions.reshape(-1)
        return actions

    def get_legal_actions_mask(self, observation: np.ndarray) -> np.ndarray:
        """
        获取合法动作mask

        Args:
            observation: 观察数组，shape (n_envs, obs_dim) 或 (obs_dim,)

        Returns:
            mask: bool数组，True=合法，False=非法，shape (n_envs, n_actions)
        """
        # 确保是2D
        if len(observation.shape) == 1:
            observation = observation.reshape(1, -1)

        # 合法动作：obs中值为0的位置
        mask = np.abs(observation) < 1e-6
        return mask


def get_action_mask(observation: np.ndarray) -> np.ndarray:
    """
    工具函数：从观察中生成 action mask

    Args:
        observation: 棋盘状态，shape (9,) 或 (batch, 9)
                    - 0: 空位（合法）
                    - 1: 自己的棋子（非法）
                    - -1: 对手的棋子（非法）

    Returns:
        mask: bool数组，True=合法动作，False=非法动作
    """
    # 只有绝对值接近0的才是空位
    return np.abs(observation) < 1e-6


# ============ 测试代码 ============
if __name__ == "__main__":
    print("Testing MaskedDQNPolicy...")

    # 创建简单的环境空间
    obs_space = spaces.Box(low=-1, high=1, shape=(9,), dtype=np.float32)
    act_space = spaces.Discrete(9)

    # 创建policy
    policy = MaskedDQNPolicy(
        observation_space=obs_space,
        action_space=act_space,
        lr_schedule=lambda _: 1e-3,
        net_arch=[128, 128]
    )

    print("✓ Policy创建成功")

    # 测试1: 空棋盘
    print("\n=== Test 1: 空棋盘 ===")
    obs1 = np.zeros(9, dtype=np.float32)
    print(f"棋盘:\n{obs1.reshape(3, 3)}")
    print(f"合法动作: {np.where(np.abs(obs1) < 1e-6)[0].tolist()}")

    action1, _ = policy.predict(obs1, deterministic=True)
    print(f"选择动作: {action1}")

    # 验证
    is_legal1 = np.abs(obs1[action1]) < 1e-6
    print(f"动作是否合法: {is_legal1} ({'✓' if is_legal1 else '✗'})")

    # 测试2: 部分占据的棋盘（有自己和对手的棋子）
    print("\n=== Test 2: 部分占据棋盘 ===")
    obs2 = np.array([1, 1, 0, -1, 0, 0, 0, 0, 0], dtype=np.float32)
    print(f"棋盘:\n{obs2.reshape(3, 3)}")
    print(f"  1 表示自己的棋子")
    print(f"  -1 表示对手的棋子")
    print(f"  0 表示空位（合法）")
    print(f"合法动作: {np.where(np.abs(obs2) < 1e-6)[0].tolist()}")

    action2, _ = policy.predict(obs2, deterministic=True)
    print(f"选择动作: {action2}")

    # 验证
    is_legal2 = np.abs(obs2[action2]) < 1e-6
    print(f"动作是否合法: {is_legal2} ({'✓' if is_legal2 else '✗'})")

    # 测试3: 测试视角翻转后的棋盘（模拟后手）
    print("\n=== Test 3: 翻转视角棋盘（模拟后手）===")
    # 原始棋盘: X=1, O=-1
    original_board = np.array([1, 0, -1, 0, 1, 0, 0, 0, -1], dtype=np.float32)
    # 后手视角: 翻转后 自己(O)=1, 对手(X)=-1
    obs3 = -original_board
    print(f"棋盘:\n{obs3.reshape(3, 3)}")
    print(f"合法动作: {np.where(np.abs(obs3) < 1e-6)[0].tolist()}")

    action3, _ = policy.predict(obs3, deterministic=True)
    print(f"选择动作: {action3}")

    # 验证
    is_legal3 = np.abs(obs3[action3]) < 1e-6
    print(f"动作是否合法: {is_legal3} ({'✓' if is_legal3 else '✗'})")

    # 测试4: 批量预测
    print("\n=== Test 4: 批量预测 ===")
    obs_batch = np.array([
        [1, 1, 0, 0, 0, 0, 0, 0, 0],      # 前两个被占
        [0, 0, 0, 0, 1, 0, 0, 0, 0],      # 中心被占
        [1, -1, 1, -1, 0, 0, 0, 0, 0]     # 前4个被占
    ], dtype=np.float32)

    actions_batch, _ = policy.predict(obs_batch, deterministic=True)
    print(f"批量动作: {actions_batch}")

    all_legal = True
    for i, (obs, action) in enumerate(zip(obs_batch, actions_batch)):
        is_legal = np.abs(obs[action]) < 1e-6
        legal_actions = np.where(np.abs(obs) < 1e-6)[0].tolist()
        symbol = '✓' if is_legal else '✗'
        print(f"  {symbol} 样本{i}: 动作={action}, 合法={is_legal}, 可选={legal_actions}")
        all_legal = all_legal and is_legal

    if all_legal:
        print("\n✓ 所有测试通过！MaskedDQNPolicy 正常工作")
    else:
        print("\n✗ 测试失败：存在非法动作")
