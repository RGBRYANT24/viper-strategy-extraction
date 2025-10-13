"""
TicTacToe自我对弈环境
智能体与自己的旧版本对战
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import copy


class TicTacToeSelfPlayEnv(gym.Env):
    """
    Tic-Tac-Toe 自我对弈环境

    智能体X与自己的旧策略O对战
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, opponent_policy=None):
        """
        Args:
            opponent_policy: 对手策略（模型），None表示随机对手
        """
        super().__init__()

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(9,),
            dtype=np.float32
        )

        self.action_space = spaces.Discrete(9)

        self.board = None
        self.done = False
        self.winner = None

        # 对手策略（可以动态更新）
        self.opponent_policy = opponent_policy

        self.win_combinations = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # 行
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # 列
            [0, 4, 8], [2, 4, 6]              # 对角线
        ]

        # 统计
        self.step_count = 0
        self._debug_print_interval = 10000

    def set_opponent_policy(self, policy):
        """更新对手策略（用于自我对弈）"""
        self.opponent_policy = policy

    def reset(self, seed=None, options=None):
        """重置环境到初始状态"""
        if seed is not None:
            np.random.seed(seed)

        self.board = np.zeros(9, dtype=np.float32)
        self.done = False
        self.winner = None
        return self.board.copy(), {}

    def step(self, action):
        """执行一步动作"""
        self.step_count += 1
        if self.step_count % self._debug_print_interval == 0:
            print(f"[SELFPLAY ENV] Step {self.step_count}")

        if self.done:
            return self.board.copy(), 0, True, False, {'error': 'game_already_done'}

        # 检查动作是否合法
        if not self._is_valid_action(action):
            self.done = True
            return self.board.copy(), -10, True, False, {'illegal_move': True}

        # 玩家 X 落子
        self.board[action] = 1

        # 检查玩家是否获胜
        if self._check_winner(1):
            self.done = True
            self.winner = 1
            return self.board.copy(), 1, True, False, {'winner': 'X'}

        # 检查是否平局
        if not self._has_empty_cells():
            self.done = True
            return self.board.copy(), 0, True, False, {'draw': True}

        # 对手O落子（使用策略或随机）
        opponent_action = self._opponent_move()
        self.board[opponent_action] = -1

        # 检查对手是否获胜
        if self._check_winner(-1):
            self.done = True
            self.winner = -1
            return self.board.copy(), -1, True, False, {'winner': 'O'}

        # 再次检查平局
        if not self._has_empty_cells():
            self.done = True
            return self.board.copy(), 0, True, False, {'draw': True}

        # 游戏继续
        return self.board.copy(), 0, False, False, {}

    def _is_valid_action(self, action):
        """检查动作是否合法"""
        return 0 <= action < 9 and self.board[action] == 0

    def _check_winner(self, player):
        """检查某个玩家是否获胜"""
        for combo in self.win_combinations:
            if all(self.board[pos] == player for pos in combo):
                return True
        return False

    def _has_empty_cells(self):
        """检查是否还有空位"""
        return np.any(self.board == 0)

    def _get_legal_actions(self):
        """获取所有合法动作"""
        return np.where(self.board == 0)[0]

    def _opponent_move(self):
        """对手选择动作"""
        if self.opponent_policy is not None:
            # 使用策略网络
            try:
                # 将视角转换为对手视角（翻转1和-1）
                opponent_view = -self.board
                action, _ = self.opponent_policy.predict(opponent_view, deterministic=False)

                # 验证动作合法性
                if self._is_valid_action(action):
                    return action
            except:
                pass  # 如果策略失败，回退到随机

        # 随机策略（或策略失败时的回退）
        legal_actions = self._get_legal_actions()
        return np.random.choice(legal_actions)

    def render(self, mode='human'):
        """渲染当前棋盘状态"""
        symbols = {1: 'X', -1: 'O', 0: '.'}
        board_2d = self.board.reshape(3, 3)

        output = "\n"
        for i in range(3):
            row = " | ".join([symbols[int(cell)] for cell in board_2d[i]])
            output += f"  {row}\n"
            if i < 2:
                output += " -----------\n"

        if mode == 'human':
            print(output)
        return output

    def close(self):
        """清理资源"""
        pass


if __name__ == "__main__":
    print("Testing TicTacToe Self-Play Environment...")

    env = TicTacToeSelfPlayEnv()

    # 测试基本功能
    print("\n=== Test: Basic Game ===")
    obs, info = env.reset()
    print(f"Initial state: {obs}")
    env.render()

    terminated = False
    step_count = 0

    while not terminated and step_count < 10:
        legal_actions = np.where(obs == 0)[0]
        if len(legal_actions) == 0:
            break
        action = np.random.choice(legal_actions)

        obs, reward, terminated, truncated, info = env.step(action)
        print(f"\nStep {step_count + 1}: action={action}, reward={reward}")
        env.render()

        step_count += 1

    print(f"\nGame ended - Info: {info}")
