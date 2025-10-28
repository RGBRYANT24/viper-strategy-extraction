"""
TicTacToe Delta-Uniform Self-Play 环境
支持从对手池中均匀采样，实现多样化的自我对弈训练
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random


class TicTacToeDeltaSelfPlayEnv(gym.Env):
    """
    Tic-Tac-Toe Delta-Uniform Self-Play 环境

    关键特性:
    1. 维护两个对手池: 基准池 (MinMax/Random) 和 学习池 (历史策略快照)
    2. 每次 reset() 时从对手池中均匀采样一个对手
    3. 支持先手/后手训练，并通过翻转棋盘保持神经网络输入一致性
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, baseline_pool=None, learned_pool=None,
                 play_as_o_prob=0.5, sampling_strategy='uniform'):
        """
        Args:
            baseline_pool: 基准策略池 (list of BasePolicy)
            learned_pool: 学习策略池 (deque of DQN policies)
            play_as_o_prob: 作为后手(O)的概率，默认0.5 (先后手各50%)
            sampling_strategy: 采样策略 ('uniform' 或 'delta-weighted')
        """
        super().__init__()

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(9,),
            dtype=np.float32
        )

        self.action_space = spaces.Discrete(9)

        # 游戏状态
        self.board = None
        self.done = False
        self.winner = None

        # 对手池
        self.baseline_pool = baseline_pool if baseline_pool is not None else []
        self.learned_pool = learned_pool  # deque，会自动更新

        # 当前对手和角色
        self.current_opponent = None
        self.play_as_o = False  # True=后手O, False=先手X
        self.play_as_o_prob = play_as_o_prob

        # 采样策略
        self.sampling_strategy = sampling_strategy

        # 获胜组合
        self.win_combinations = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # 行
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # 列
            [0, 4, 8], [2, 4, 6]              # 对角线
        ]

        # 统计
        self.step_count = 0
        self.episode_count = 0
        self._debug_print_interval = 10000

    def set_opponent_pools(self, baseline_pool=None, learned_pool=None):
        """更新对手池（用于外部动态更新）"""
        if baseline_pool is not None:
            self.baseline_pool = baseline_pool
        if learned_pool is not None:
            self.learned_pool = learned_pool

    def _sample_opponent(self):
        """从对手池中采样一个对手"""
        # 合并两个池
        all_opponents = list(self.baseline_pool)  # 基准策略

        if self.learned_pool is not None and len(self.learned_pool) > 0:
            all_opponents.extend(list(self.learned_pool))  # 学习策略

        if len(all_opponents) == 0:
            # 如果没有对手，返回None（将使用随机策略）
            return None

        if self.sampling_strategy == 'uniform':
            # 均匀采样
            return random.choice(all_opponents)
        elif self.sampling_strategy == 'delta-weighted':
            # TODO: 实现基于胜率的 delta-weighted 采样
            # 需要维护每个对手的胜率统计
            return random.choice(all_opponents)
        else:
            return random.choice(all_opponents)

    def reset(self, seed=None, options=None):
        """重置环境到初始状态"""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.board = np.zeros(9, dtype=np.float32)
        self.done = False
        self.winner = None
        self.episode_count += 1

        # 随机决定先手/后手
        self.play_as_o = (np.random.random() < self.play_as_o_prob)

        # 采样对手
        self.current_opponent = self._sample_opponent()

        # 如果是后手(O)，对手先走
        if self.play_as_o:
            opponent_action = self._opponent_move()
            if opponent_action is not None:
                self.board[opponent_action] = 1  # 对手下X (在实际棋盘上)

        # 返回观察（总是从自己的视角）
        obs = self._get_observation()
        return obs, {}

    def step(self, action):
        """执行一步动作"""
        self.step_count += 1
        if self.step_count % self._debug_print_interval == 0:
            learned_count = len(self.learned_pool) if self.learned_pool else 0
            print(f"[DELTA-SELFPLAY] Step {self.step_count}, "
                  f"Episodes {self.episode_count}, "
                  f"Pool: {len(self.baseline_pool)} baseline + {learned_count} learned")

        if self.done:
            return self._get_observation(), 0, True, False, {'error': 'game_already_done'}

        # 检查动作是否合法
        if not self._is_valid_action(action):
            self.done = True
            return self._get_observation(), -10, True, False, {'illegal_move': True}

        # 我方落子
        # 在实际棋盘上：先手X=1, 后手O=-1
        my_marker = -1 if self.play_as_o else 1
        self.board[action] = my_marker

        # 检查我方是否获胜
        if self._check_winner(my_marker):
            self.done = True
            self.winner = my_marker
            return self._get_observation(), 1, True, False, {'winner': 'self'}

        # 检查是否平局
        if not self._has_empty_cells():
            self.done = True
            return self._get_observation(), 0, True, False, {'draw': True}

        # 对手落子
        opponent_action = self._opponent_move()
        if opponent_action is not None:
            opponent_marker = 1 if self.play_as_o else -1
            self.board[opponent_action] = opponent_marker

            # 检查对手是否获胜
            if self._check_winner(opponent_marker):
                self.done = True
                self.winner = opponent_marker
                return self._get_observation(), -1, True, False, {'winner': 'opponent'}

            # 再次检查平局
            if not self._has_empty_cells():
                self.done = True
                return self._get_observation(), 0, True, False, {'draw': True}

        # 游戏继续
        return self._get_observation(), 0, False, False, {}

    def _get_observation(self):
        """
        获取观察（从自己的视角）

        关键：无论先手后手，神经网络输入必须一致
        - 自己的棋子总是表示为 1
        - 对手的棋子总是表示为 -1
        """
        if self.play_as_o:
            # 如果是后手O，翻转棋盘视角
            # 实际棋盘: X=1, O=-1
            # 网络输入: 自己(O)=1, 对手(X)=-1
            return -self.board.copy()
        else:
            # 如果是先手X，直接返回
            # 实际棋盘: X=1, O=-1
            # 网络输入: 自己(X)=1, 对手(O)=-1
            return self.board.copy()

    def _is_valid_action(self, action):
        """检查动作是否合法（基于实际棋盘）"""
        return 0 <= action < 9 and self.board[action] == 0

    def _check_winner(self, player):
        """检查某个玩家是否获胜（基于实际棋盘）"""
        for combo in self.win_combinations:
            if all(self.board[pos] == player for pos in combo):
                return True
        return False

    def _has_empty_cells(self):
        """检查是否还有空位"""
        return np.any(self.board == 0)

    def _get_legal_actions(self):
        """获取所有合法动作（基于实际棋盘）"""
        return np.where(self.board == 0)[0]

    def _opponent_move(self):
        """对手选择动作"""
        if self.current_opponent is not None:
            try:
                # 将视角转换为对手视角
                # 对手看到的棋盘：自己=1，对手=-1
                opponent_view = -self.board

                # 计算对手的 action mask（从对手视角）
                # 空位置是合法动作
                action_mask = (self.board == 0).astype(np.int8)

                # 调用对手策略（尝试传递 action_masks）
                # 如果策略支持 action_masks 参数（如 MaskablePPO），则使用
                # 否则回退到不带 mask 的调用
                try:
                    action, _ = self.current_opponent.predict(
                        opponent_view,
                        deterministic=False,
                        action_masks=action_mask
                    )
                except TypeError:
                    # 策略不支持 action_masks 参数（如 RandomPolicy）
                    action, _ = self.current_opponent.predict(opponent_view, deterministic=False)

                # 验证动作合法性（基于实际棋盘）
                if self._is_valid_action(action):
                    return action
                else:
                    # 如果策略返回非法动作，打印警告并回退到随机
                    if self.step_count <= 10:
                        print(f"[WARNING] Opponent returned illegal action {action}, falling back to random")
            except Exception as e:
                # 如果策略失败，回退到随机
                if self.step_count <= 10:
                    print(f"[WARNING] Opponent policy failed: {e}, falling back to random")

        # 随机策略（或策略失败时的回退）
        legal_actions = self._get_legal_actions()
        if len(legal_actions) > 0:
            return np.random.choice(legal_actions)
        return None

    def render(self, mode='human'):
        """渲染当前棋盘状态"""
        symbols = {1: 'X', -1: 'O', 0: '.'}
        board_2d = self.board.reshape(3, 3)

        output = "\n"
        output += f"  Playing as: {'O (后手)' if self.play_as_o else 'X (先手)'}\n"
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


# ============ 测试代码 ============
if __name__ == "__main__":
    print("Testing TicTacToe Delta-Uniform Self-Play Environment...")

    # 创建基准策略
    from gymnasium import spaces
    from gym_env.policies import RandomPlayerPolicy, MinMaxPlayerPolicy

    obs_space = spaces.Box(low=-1, high=1, shape=(9,), dtype=np.float32)
    act_space = spaces.Discrete(9)

    baseline_pool = [
        RandomPlayerPolicy(obs_space, act_space),
        MinMaxPlayerPolicy(obs_space, act_space)
    ]

    # 创建环境
    env = TicTacToeDeltaSelfPlayEnv(
        baseline_pool=baseline_pool,
        learned_pool=None,
        play_as_o_prob=0.5
    )

    print("\n=== Test 1: 先手(X) vs 随机对手 ===")
    obs, info = env.reset()
    print(f"Initial observation (from agent's view):\n{obs.reshape(3, 3)}")
    env.render()

    # 测试几步
    for step in range(3):
        legal_actions = np.where(obs == 0)[0]
        if len(legal_actions) == 0:
            break
        action = np.random.choice(legal_actions)

        obs, reward, terminated, truncated, info = env.step(action)
        print(f"\nStep {step + 1}: action={action}, reward={reward}")
        env.render()

        if terminated:
            print(f"Game ended: {info}")
            break

    print("\n=== Test 2: 后手(O) vs MinMax ===")
    obs, info = env.reset()
    print(f"Playing as: {'O' if env.play_as_o else 'X'}")
    print(f"Initial observation (from agent's view):\n{obs.reshape(3, 3)}")
    env.render()

    print("\n✓ Delta-Uniform Self-Play Environment 测试完成")
