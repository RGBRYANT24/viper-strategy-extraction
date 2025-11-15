"""
TicTacToe 环境实现 - 与 VIPER 框架完全兼容
文件位置: envs/tictactoe.py

使用方式：
1. 将此文件放入 viper-verifiable-rl-impl/envs/ 目录
2. 在 main.py 中注册环境
3. 使用框架的标准命令训练
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gym_env.policies.baseline_policies import MinMaxPlayerPolicy, RandomPlayerPolicy


class TicTacToeEnv(gym.Env):
    """
    Tic-Tac-Toe 单人训练环境（对战随机对手）
    
    状态表示: 9维向量 [pos0, pos1, ..., pos8]
        - 1: X (我方)
        - -1: O (对手)  
        - 0: 空
    
    动作空间: Discrete(9) - 选择 0-8 号位置落子
    
    奖励:
        - +1: 获胜
        - -1: 失败
        - 0: 平局
        - -10: 非法移动
    """
    
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, opponent_type='random', minmax_depth=9, play_as_o_prob=0.5):
        """
        Args:
            opponent_type: 对手类型，可选 'random' 或 'minmax'
            minmax_depth: MinMax搜索深度（仅当opponent_type='minmax'时有效）
            play_as_o_prob: 是玩家作为后手(O)的概率
                        - 0.0: 总是先手(X)
                        - 0.5: 随机先后手 (默认)
                        - 1.0: 总是后手(O)
                        注意这里是玩家(神经网络 决策树等)作为先手还是后手
                        神经网络看到棋盘永远是1,X代表自己, -1,O代表对手
                        
        """
        super().__init__()

        # 状态空间: 9个位置，每个位置 -1, 0, 或 1
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(9,),
            dtype=np.float32
        )

        # 先后手设置
        self.play_as_o_prob = play_as_o_prob
        self.play_as_o = False # 当前是否后手在reset的时候决定

        # 动作空间: 9个可能的位置
        self.action_space = spaces.Discrete(9)

        # 游戏状态
        self.board = None
        self.done = False
        self.winner = None

        # 对手设置 - 使用 baseline_policies
        self.opponent_type = opponent_type
        if opponent_type == 'minmax':
            self.opponent_policy = MinMaxPlayerPolicy(
                self.observation_space,
                self.action_space,
                depth=minmax_depth
            )
        else:  # random
            self.opponent_policy = RandomPlayerPolicy(
                self.observation_space,
                self.action_space
            )

        # 获胜组合（行、列、对角线）
        self.win_combinations = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # 行
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # 列
            [0, 4, 8], [2, 4, 6]              # 对角线
        ]

        # 调试计数器
        self.step_count = 0
        self._debug_print_interval = 100
        
    def reset(self, seed=None, options=None):
        """重置环境到初始状态"""
        if seed is not None:
            np.random.seed(seed)

        self.board = np.zeros(9, dtype=np.float32)
        self.done = False
        self.winner = None

        # 随机决定先手/后手
        self.play_as_o = np.random.rand() < self.play_as_o_prob

        # 如果玩家作为后手，先让环境下第一步
        if self.play_as_o:
            oppenent_action = self._opponent_move()
            if oppenent_action is not None:
                self.board[oppenent_action] = 1 # 实际棋盘上对手是1

        # return self.board.copy(), {}
        # 这里要做视角转换 根据先后手
        return self._get_observation(), {}
    
    def _get_observation(self):
        """获取当前棋盘状态"""
        if self.play_as_o:
            # 玩家作为O，视角转换
            obs = -self.board.copy()
        else:
            obs = self.board.copy()
        return obs

    def step(self, action):
        """
        执行一步动作

        Args:
            action: 0-8 的整数，表示落子位置

        Returns:
            observation: 新的棋盘状态
            reward: 奖励值
            terminated: 是否因游戏结束而终止
            truncated: 是否因时间限制而截断（总是False）
            info: 额外信息
        """
        # 调试输出
        self.step_count += 1
        if self.step_count % self._debug_print_interval == 0:
            print(f"[ENV DEBUG] Step {self.step_count} (opponent: {self.opponent_type})")

        if self.done:
            return self._get_observation(), 0, True, False, {'error': 'game_already_done'}

        # 检查动作是否合法
        if not self._is_valid_action(action):
            # 非法移动，游戏结束并给予惩罚
            self.done = True
            return self._get_observation(), -10, True, False, {'illegal_move': True}

        my_marker = -1 if self.play_as_o else 1

        # 玩家 落子
        self.board[action] = my_marker

        # 检查玩家是否获胜
        if self._check_winner(my_marker):
            self.done = True
            self.winner = my_marker
            return self._get_observation(), 1, True, False, {'winner': 'self'}

        # 检查是否平局
        if not self._has_empty_cells():
            self.done = True
            return self._get_observation(), 0, True, False, {'draw': True}

        # 对手 O 随机落子
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
        """
        对手选择动作 - 使用 baseline_policies
        
        关键：对手也需要从自己的视角看棋盘
        无论对手是X还是O，都要看到"自己=1，对手=-1"
        因为调用的是 baseline_policies 中的 MinMaxPlayerPolicy 其中predict方法认为自己play的是1
        """
        opponent_marker = 1 if self.play_as_o else -1
        if opponent_marker == 1:
            # 对手(环境)是X(先手)，物理标记=1，不需要转换
            opponent_view = self.board.copy()
        else:
            # 对手是O(后手)，物理标记=-1，需要转换让其看到自己=1
            opponent_view = -self.board.copy()
        # 使用baselinee_policies中的predict方法选择动作
        try:
            action, _ = self.opponent_policy.predict(opponent_view)
            if self._is_valid_action(action):
                return action
        except Exception as e:
            raise RuntimeError(f"Opponent policy Minmax failed to predict action: {e}")
        # 如果策略失败或动作非法，随机选择
        legal_actions = self._get_legal_actions()
        if len(legal_actions) == 0:
            return None
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


class TicTacToeSymmetricEnv(TicTacToeEnv):
    """
    增强版 TicTacToe 环境 - 支持数据增强
    通过旋转和镜像增加训练数据多样性
    """
    
    def __init__(self, use_augmentation=True):
        super().__init__()
        self.use_augmentation = use_augmentation
        
    def step(self, action):
        """支持数据增强的 step"""
        obs, reward, done, info = super().step(action)
        
        # 如果启用数据增强，随机应用旋转或镜像
        if self.use_augmentation and not done:
            if np.random.random() < 0.2:  # 20% 概率
                obs = self._apply_random_transform(obs)
        
        return obs, reward, done, info
    
    def _apply_random_transform(self, board):
        """应用随机的对称变换"""
        board_2d = board.reshape(3, 3)
        
        # 随机选择变换
        transform = np.random.choice(['rotate90', 'rotate180', 'rotate270', 'flip_h', 'flip_v'])
        
        if transform == 'rotate90':
            board_2d = np.rot90(board_2d, k=1)
        elif transform == 'rotate180':
            board_2d = np.rot90(board_2d, k=2)
        elif transform == 'rotate270':
            board_2d = np.rot90(board_2d, k=3)
        elif transform == 'flip_h':
            board_2d = np.fliplr(board_2d)
        elif transform == 'flip_v':
            board_2d = np.flipud(board_2d)
        
        return board_2d.flatten()


# ============ Gym 注册 ============
# 注册已经在 gym_env/__init__.py 中完成

# ============ 测试代码 ============
if __name__ == "__main__":
    print("Testing TicTacToe Environment...")
    
    env = TicTacToeEnv()
    
    # 测试 1: 基本功能
    print("\n=== Test 1: Basic Functionality ===")
    obs, info = env.reset()
    print(f"Initial state: {obs}")
    env.render()

    # 测试 2: 完整游戏
    print("\n=== Test 2: Full Game ===")
    obs, info = env.reset()
    terminated = False
    step_count = 0

    while not terminated and step_count < 10:
        # 随机选择合法动作
        legal_actions = np.where(obs == 0)[0]
        if len(legal_actions) == 0:
            break
        action = np.random.choice(legal_actions)

        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {step_count + 1}: action={action}, reward={reward}")
        env.render()

        step_count += 1

    print(f"\nFinal game state - Terminated: {terminated}, Steps: {step_count}")