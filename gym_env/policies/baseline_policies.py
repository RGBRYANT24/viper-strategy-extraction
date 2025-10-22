"""
基准策略包装器 - 用于 Delta-Uniform Self-Play
提供与 stable-baselines3 BasePolicy 兼容的 MinMax 和 Random 策略
"""

import numpy as np
from stable_baselines3.common.policies import BasePolicy
from gymnasium import spaces


class RandomPlayerPolicy(BasePolicy):
    """
    随机策略 - 从所有合法动作中随机选择
    包装为 BasePolicy 接口以兼容 DQN 策略池
    """

    def __init__(self, observation_space, action_space):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space
        )

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        """
        预测动作（兼容 stable-baselines3 接口）

        Args:
            observation: numpy数组，形状为 (9,) 或 (batch_size, 9)
            state: RNN状态（未使用）
            episode_start: 是否新回合开始（未使用）
            deterministic: 是否确定性（对随机策略无影响）

        Returns:
            action: 选择的动作
            state: 更新的状态（None）
        """
        # 处理批量输入
        if len(observation.shape) == 1:
            observation = observation.reshape(1, -1)
            single_obs = True
        else:
            single_obs = False

        actions = []
        for obs in observation:
            # 获取合法动作（空位）
            legal_actions = np.where(obs == 0)[0]

            if len(legal_actions) > 0:
                action = np.random.choice(legal_actions)
            else:
                # 没有合法动作，随机返回
                action = np.random.randint(0, 9)

            actions.append(action)

        actions = np.array(actions)

        if single_obs:
            return actions[0], None
        return actions, None

    def _predict(self, observation, deterministic=True):
        """内部预测方法"""
        action, _ = self.predict(observation, deterministic=deterministic)
        return action

    def forward(self, obs, deterministic=True):
        """前向传播（兼容接口）"""
        return self._predict(obs, deterministic)


class MinMaxPlayerPolicy(BasePolicy):
    """
    MinMax 策略 - 使用 Minimax 算法选择最优动作
    包装为 BasePolicy 接口以兼容 DQN 策略池
    """

    def __init__(self, observation_space, action_space, depth=9):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space
        )
        self.depth = depth

        # 获胜组合
        self.win_combinations = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # 行
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # 列
            [0, 4, 8], [2, 4, 6]              # 对角线
        ]

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        """
        使用 MinMax 算法预测最优动作（兼容 stable-baselines3 接口）

        Args:
            observation: numpy数组，形状为 (9,) 或 (batch_size, 9)
            state: RNN状态（未使用）
            episode_start: 是否新回合开始（未使用）
            deterministic: 是否确定性

        Returns:
            action: 选择的动作
            state: 更新的状态（None）
        """
        # 处理批量输入
        if len(observation.shape) == 1:
            observation = observation.reshape(1, -1)
            single_obs = True
        else:
            single_obs = False

        actions = []
        for board in observation:
            action = self._minmax_move(board.copy(), player=1)
            actions.append(action)

        actions = np.array(actions)

        if single_obs:
            return actions[0], None
        return actions, None

    def _predict(self, observation, deterministic=True):
        """内部预测方法"""
        action, _ = self.predict(observation, deterministic=deterministic)
        return action

    def _minmax_move(self, board, player):
        """使用MinMax算法选择最优动作"""
        best_score = float('-inf')
        best_action = None
        legal_actions = np.where(board == 0)[0]

        if len(legal_actions) == 0:
            return 0

        # 早期游戏启发式优化
        if len(legal_actions) == 9:  # 第一步
            return 4  # 中心位置
        elif len(legal_actions) >= 7:  # 前两步
            if 4 in legal_actions:
                return 4  # 优先中心
            corners = [0, 2, 6, 8]
            available_corners = [c for c in corners if c in legal_actions]
            if available_corners:
                return available_corners[0]

        for action in legal_actions:
            board[action] = player
            score = self._minimax(board, 0, False, player, float('-inf'), float('inf'))
            board[action] = 0

            if score > best_score:
                best_score = score
                best_action = action

            # 如果找到必胜策略，直接返回
            if best_score >= 10:
                break

        return best_action if best_action is not None else legal_actions[0]

    def _minimax(self, board, depth, is_maximizing, player, alpha, beta):
        """MinMax算法核心（带Alpha-Beta剪枝）"""
        winner = self._check_winner_state(board)
        if winner == player:
            return 10 - depth  # 越快赢越好
        elif winner == -player:
            return -10 + depth  # 越晚输越好
        elif winner == 0:  # 平局
            return 0

        if depth >= self.depth:
            return 0

        legal_actions = np.where(board == 0)[0]
        if len(legal_actions) == 0:
            return 0

        if is_maximizing:
            max_eval = float('-inf')
            for action in legal_actions:
                board[action] = player
                eval_score = self._minimax(board, depth + 1, False, player, alpha, beta)
                board[action] = 0
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Beta剪枝
            return max_eval
        else:
            min_eval = float('inf')
            for action in legal_actions:
                board[action] = -player
                eval_score = self._minimax(board, depth + 1, True, player, alpha, beta)
                board[action] = 0
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Alpha剪枝
            return min_eval

    def _check_winner_state(self, board):
        """
        检查游戏状态
        返回: 1=X赢, -1=O赢, 0=平局, None=游戏未结束
        """
        for combo in self.win_combinations:
            if all(board[pos] == 1 for pos in combo):
                return 1
            elif all(board[pos] == -1 for pos in combo):
                return -1

        if np.any(board == 0):
            return None  # 游戏未结束

        return 0  # 平局

    def forward(self, obs, deterministic=True):
        """前向传播（兼容接口）"""
        return self._predict(obs, deterministic)


# ============ 测试代码 ============
if __name__ == "__main__":
    print("Testing Baseline Policies...")

    # 创建空间
    obs_space = spaces.Box(low=-1, high=1, shape=(9,), dtype=np.float32)
    act_space = spaces.Discrete(9)

    # 测试 RandomPlayerPolicy
    print("\n=== Test 1: RandomPlayerPolicy ===")
    random_policy = RandomPlayerPolicy(obs_space, act_space)

    test_board = np.array([1, 0, -1, 0, 1, 0, 0, 0, -1], dtype=np.float32)
    print(f"Board:\n{test_board.reshape(3, 3)}")

    for i in range(5):
        action, _ = random_policy.predict(test_board)
        print(f"  Trial {i+1}: action={action}")

    # 测试 MinMaxPlayerPolicy
    print("\n=== Test 2: MinMaxPlayerPolicy ===")
    minmax_policy = MinMaxPlayerPolicy(obs_space, act_space)

    # 测试场景1：简单获胜机会
    board1 = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    print(f"\nBoard (should choose 2 to win):")
    print(board1.reshape(3, 3))
    action, _ = minmax_policy.predict(board1)
    print(f"Action: {action} (expected: 2)")

    # 测试场景2：防守
    board2 = np.array([-1, -1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    print(f"\nBoard (should choose 2 to block):")
    print(board2.reshape(3, 3))
    action, _ = minmax_policy.predict(board2)
    print(f"Action: {action} (expected: 2)")

    # 测试场景3：空棋盘
    board3 = np.zeros(9, dtype=np.float32)
    print(f"\nBoard (empty, should choose center=4):")
    print(board3.reshape(3, 3))
    action, _ = minmax_policy.predict(board3)
    print(f"Action: {action} (expected: 4)")

    print("\n✓ Baseline Policies 测试完成")
