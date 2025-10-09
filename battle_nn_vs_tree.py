"""
让神经网络（Oracle）和决策树（VIPER）在井字棋中对战
用于验证两个模型的策略一致性
"""

import argparse
import numpy as np
import joblib
from stable_baselines3 import PPO, DQN, A2C
from gym_env import make_env
from model.paths import get_oracle_path, get_viper_path


class TicTacToeBattleEnv:
    """
    自定义对战环境，允许两个智能体互相对战
    """
    def __init__(self):
        self.board = None
        self.current_player = None  # 1 for X, -1 for O
        self.done = False
        self.winner = None

        self.win_combinations = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # 行
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # 列
            [0, 4, 8], [2, 4, 6]              # 对角线
        ]

    def reset(self):
        """重置棋盘"""
        self.board = np.zeros(9, dtype=np.float32)
        self.current_player = 1  # X先手
        self.done = False
        self.winner = None
        return self.board.copy()

    def get_legal_actions(self):
        """获取合法动作"""
        return np.where(self.board == 0)[0]

    def step(self, action):
        """执行一步"""
        if self.done:
            return self.board.copy(), 0, True, {'error': 'game_already_done'}

        # 检查动作合法性
        if action < 0 or action >= 9 or self.board[action] != 0:
            # 非法移动，当前玩家输
            self.done = True
            self.winner = -self.current_player
            return self.board.copy(), -10, True, {'illegal_move': True, 'player': self.current_player}

        # 落子
        self.board[action] = self.current_player

        # 检查是否获胜
        if self._check_winner(self.current_player):
            self.done = True
            self.winner = self.current_player
            reward = 1 if self.current_player == 1 else -1
            return self.board.copy(), reward, True, {'winner': self.current_player}

        # 检查平局
        if not self._has_empty_cells():
            self.done = True
            self.winner = 0
            return self.board.copy(), 0, True, {'draw': True}

        # 切换玩家
        self.current_player = -self.current_player
        return self.board.copy(), 0, False, {}

    def _check_winner(self, player):
        """检查某个玩家是否获胜"""
        for combo in self.win_combinations:
            if all(self.board[pos] == player for pos in combo):
                return True
        return False

    def _has_empty_cells(self):
        """检查是否还有空位"""
        return np.any(self.board == 0)

    def render(self):
        """渲染棋盘"""
        symbols = {1: 'X', -1: 'O', 0: '.'}
        board_2d = self.board.reshape(3, 3)

        print()
        for i in range(3):
            row = " | ".join([symbols[int(cell)] for cell in board_2d[i]])
            print(f"  {row}")
            if i < 2:
                print(" -----------")
        print()


class MinMaxPlayer:
    """MinMax算法玩家（最优策略）"""
    def __init__(self, player_id=1, depth_limit=9):
        """
        Args:
            player_id: 玩家标识 (1 for X, -1 for O)
            depth_limit: 搜索深度限制
        """
        self.player_id = player_id
        self.depth_limit = depth_limit
        self.win_combinations = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # 行
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # 列
            [0, 4, 8], [2, 4, 6]              # 对角线
        ]
        print(f"已初始化MinMax玩家 (depth={depth_limit})")

    def predict(self, obs):
        """使用MinMax算法预测最优动作"""
        # 根据当前棋盘状态推断当前玩家
        num_x = np.sum(obs == 1)
        num_o = np.sum(obs == -1)
        current_player = 1 if num_x == num_o else -1

        best_action = self._minimax_search(obs.copy(), current_player)
        return best_action

    def _minimax_search(self, board, player):
        """MinMax搜索入口"""
        best_score = float('-inf')
        best_action = None
        legal_actions = np.where(board == 0)[0]

        if len(legal_actions) == 0:
            # 如果没有合法动作，返回任意位置
            return 0

        for action in legal_actions:
            # 尝试这个动作
            board[action] = player
            score = self._minimax(board, 0, False, player, float('-inf'), float('inf'))
            board[action] = 0  # 撤销

            if score > best_score:
                best_score = score
                best_action = action

        return best_action if best_action is not None else legal_actions[0]

    def _minimax(self, board, depth, is_maximizing, player, alpha, beta):
        """MinMax算法核心（带Alpha-Beta剪枝）"""
        # 检查终止条件
        winner = self._check_winner_state(board)
        if winner == player:
            return 10 - depth  # 越快赢越好
        elif winner == -player:
            return -10 + depth  # 越晚输越好
        elif winner == 0:  # 平局
            return 0

        # 检查深度限制
        if depth >= self.depth_limit:
            return 0

        legal_actions = np.where(board == 0)[0]
        if len(legal_actions) == 0:
            return 0  # 平局

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
        # 检查所有获胜组合
        for combo in self.win_combinations:
            if all(board[pos] == 1 for pos in combo):
                return 1
            elif all(board[pos] == -1 for pos in combo):
                return -1

        # 检查是否还有空位
        if np.any(board == 0):
            return None  # 游戏未结束

        return 0  # 平局


class DecisionTreePlayer:
    """决策树玩家"""
    def __init__(self, model_path, debug=False):
        self.model = joblib.load(model_path)
        self.debug = debug
        self.predict_count = 0
        print(f"已加载决策树模型: {model_path}")
        print(f"  模型类型: {type(self.model)}")

        # 检查是否有内部tree
        if hasattr(self.model, 'tree'):
            print(f"  这是TreeWrapper，内部树类型: {type(self.model.tree)}")
            if hasattr(self.model.tree, 'n_classes_'):
                print(f"  类别数: {self.model.tree.n_classes_}")
                print(f"  类别: {self.model.tree.classes_}")

    def predict(self, obs):
        """预测动作"""
        obs_reshaped = obs.reshape(1, -1)
        action = self.model.predict(obs_reshaped)[0]

        # 调试输出前几次预测
        self.predict_count += 1
        if self.debug and self.predict_count <= 5:
            print(f"\n[TREE DEBUG {self.predict_count}]")
            print(f"  输入棋盘: {obs}")
            print(f"  预测动作: {action}, 类型: {type(action)}")
            print(f"  合法动作: {np.where(obs == 0)[0]}")
            print(f"  动作是否合法: {action in np.where(obs == 0)[0]}")

        return action


class NeuralNetPlayer:
    """神经网络玩家"""
    def __init__(self, model_path, model_type='auto'):
        """
        Args:
            model_path: 模型文件路径
            model_type: 模型类型 ('auto', 'PPO', 'DQN', 'A2C')
                       'auto' 会自动尝试检测模型类型
        """
        self.model = None
        self.model_type = model_type

        if model_type == 'auto':
            # 自动检测模型类型
            self.model, self.model_type = self._auto_load(model_path)
        else:
            # 手动指定类型
            if model_type == 'PPO':
                self.model = PPO.load(model_path)
            elif model_type == 'DQN':
                self.model = DQN.load(model_path)
            elif model_type == 'A2C':
                self.model = A2C.load(model_path)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

        print(f"已加载神经网络模型 ({self.model_type}): {model_path}")

    def _auto_load(self, model_path):
        """自动检测并加载模型"""
        # 按常见度顺序尝试不同的模型类型
        model_classes = [
            ('DQN', DQN),
            ('PPO', PPO),
            ('A2C', A2C),
        ]

        last_error = None
        for model_name, model_class in model_classes:
            try:
                model = model_class.load(model_path)
                print(f"✓ 自动检测到模型类型: {model_name}")
                return model, model_name
            except Exception as e:
                last_error = e
                continue

        # 如果都失败了，抛出错误
        raise ValueError(f"无法加载模型 {model_path}。最后的错误: {last_error}")

    def predict(self, obs):
        """预测动作"""
        action, _ = self.model.predict(obs, deterministic=True)
        return action


def battle_two_players(player1, player2, n_games=100, verbose=False, start_player=1):
    """
    让两个玩家对战

    Args:
        player1: 玩家1（扮演X）
        player2: 玩家2（扮演O）
        n_games: 对战局数
        verbose: 是否打印详细信息
        start_player: 先手玩家 (1=player1先手, -1=player2先手)

    Returns:
        dict: 包含胜率、平局率等统计信息
    """
    env = TicTacToeBattleEnv()

    results = {
        'player1_wins': 0,    # 玩家1获胜次数
        'player2_wins': 0,    # 玩家2获胜次数
        'draws': 0,           # 平局次数
        'illegal_moves': 0,   # 非法移动次数
        'player1_illegal': 0, # 玩家1非法移动
        'player2_illegal': 0, # 玩家2非法移动
    }

    for game_idx in range(n_games):
        obs = env.reset()

        # 根据start_player决定谁先手
        if start_player == -1:
            env.current_player = -1

        if verbose:
            print(f"\n{'='*50}")
            print(f"游戏 {game_idx + 1}/{n_games}")
            print(f"{'='*50}")
            env.render()

        step_count = 0
        max_steps = 9  # 井字棋最多9步

        while not env.done and step_count < max_steps:
            # 根据当前玩家选择对应的智能体
            if env.current_player == 1:
                current_agent = player1
                agent_name = "Player1 (X)"
            else:
                current_agent = player2
                agent_name = "Player2 (O)"

            # 预测动作
            action = current_agent.predict(obs)

            if verbose:
                print(f"\n{agent_name} 选择动作: {action}")

            # 执行动作
            obs, reward, done, info = env.step(action)

            if verbose:
                env.render()

            step_count += 1

            # 检查游戏是否结束
            if done:
                if 'illegal_move' in info:
                    results['illegal_moves'] += 1
                    if info['player'] == 1:
                        results['player2_wins'] += 1
                        results['player1_illegal'] += 1
                        if verbose:
                            print(f"Player1 (X) 非法移动，Player2 (O) 获胜！")
                    else:
                        results['player1_wins'] += 1
                        results['player2_illegal'] += 1
                        if verbose:
                            print(f"Player2 (O) 非法移动，Player1 (X) 获胜！")
                elif 'winner' in info:
                    if info['winner'] == 1:
                        results['player1_wins'] += 1
                        if verbose:
                            print("Player1 (X) 获胜！")
                    else:
                        results['player2_wins'] += 1
                        if verbose:
                            print("Player2 (O) 获胜！")
                elif 'draw' in info:
                    results['draws'] += 1
                    if verbose:
                        print("平局！")
                break

    # 计算统计信息
    total_games = n_games
    results['player1_win_rate'] = results['player1_wins'] / total_games * 100
    results['player2_win_rate'] = results['player2_wins'] / total_games * 100
    results['draw_rate'] = results['draws'] / total_games * 100
    results['total_games'] = total_games

    return results


def print_battle_results(results, player1_name, player2_name):
    """打印对战结果"""
    print("\n" + "="*70)
    print(f"对战结果: {player1_name} vs {player2_name}")
    print("="*70)
    print(f"总局数: {results['total_games']}")
    print(f"\n{player1_name} (X) 获胜: {results['player1_wins']} 局 ({results['player1_win_rate']:.1f}%)")
    print(f"{player2_name} (O) 获胜: {results['player2_wins']} 局 ({results['player2_win_rate']:.1f}%)")
    print(f"平局: {results['draws']} 局 ({results['draw_rate']:.1f}%)")
    print(f"\n非法移动总数: {results['illegal_moves']}")
    print(f"  - {player1_name} 非法移动: {results['player1_illegal']}")
    print(f"  - {player2_name} 非法移动: {results['player2_illegal']}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description="神经网络 vs 决策树对战")

    # 基本参数
    parser.add_argument("--env-name", type=str, default="TicTacToe-v0",
                        help="环境名称")
    parser.add_argument("--n-games", type=int, default=100,
                        help="对战局数")
    parser.add_argument("--verbose", action='store_true',
                        help="是否打印详细对战过程")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")

    # 模型路径
    parser.add_argument("--oracle-path", type=str,
                        default="log/oracle_TicTacToe-v0.zip",
                        help="神经网络模型路径")
    parser.add_argument("--viper-path", type=str,
                        default="log/viper_TicTacToe-v0_all-leaves_10.joblib",
                        help="决策树模型路径")
    parser.add_argument("--model-type", type=str, default="auto",
                        choices=['auto', 'PPO', 'DQN', 'A2C'],
                        help="神经网络模型类型 (auto=自动检测)")

    # 对战设置
    parser.add_argument("--mode", type=str, default="all",
                        choices=['nn-vs-tree', 'tree-vs-nn', 'both',
                                 'nn-vs-minmax', 'tree-vs-minmax', 'all'],
                        help="对战模式: 两两对战或全部对战")
    parser.add_argument("--use-minmax", action='store_true',
                        help="是否包含MinMax玩家")
    parser.add_argument("--minmax-depth", type=int, default=9,
                        help="MinMax搜索深度")
    parser.add_argument("--debug", action='store_true',
                        help="启用调试输出")

    args = parser.parse_args()

    # 设置随机种子
    np.random.seed(args.seed)

    print("\n" + "="*70)
    print("井字棋对战: 神经网络 vs 决策树 vs MinMax")
    print("="*70)

    # 加载模型
    print("\n正在加载模型...")
    nn_player = NeuralNetPlayer(args.oracle_path, args.model_type)
    tree_player = DecisionTreePlayer(args.viper_path, debug=args.debug)
    minmax_player = MinMaxPlayer(depth_limit=args.minmax_depth)

    # 存储所有对战结果
    all_results = {}

    # 执行对战
    if args.mode in ['nn-vs-tree', 'both', 'all']:
        print("\n" + "="*70)
        print("对战 1: 神经网络先手 (X) vs 决策树后手 (O)")
        print("="*70)
        r1 = battle_two_players(nn_player, tree_player, n_games=args.n_games, verbose=args.verbose)
        print_battle_results(r1, "神经网络", "决策树")
        all_results['nn_vs_tree'] = r1

    if args.mode in ['tree-vs-nn', 'both', 'all']:
        print("\n" + "="*70)
        print("对战 2: 决策树先手 (X) vs 神经网络后手 (O)")
        print("="*70)
        r2 = battle_two_players(tree_player, nn_player, n_games=args.n_games, verbose=args.verbose)
        print_battle_results(r2, "决策树", "神经网络")
        all_results['tree_vs_nn'] = r2

    if args.mode in ['nn-vs-minmax', 'all']:
        print("\n" + "="*70)
        print("对战 3: 神经网络先手 (X) vs MinMax后手 (O)")
        print("="*70)
        r3 = battle_two_players(nn_player, minmax_player, n_games=args.n_games, verbose=args.verbose)
        print_battle_results(r3, "神经网络", "MinMax")
        all_results['nn_vs_minmax'] = r3

        print("\n" + "="*70)
        print("对战 4: MinMax先手 (X) vs 神经网络后手 (O)")
        print("="*70)
        r4 = battle_two_players(minmax_player, nn_player, n_games=args.n_games, verbose=args.verbose)
        print_battle_results(r4, "MinMax", "神经网络")
        all_results['minmax_vs_nn'] = r4

    if args.mode in ['tree-vs-minmax', 'all']:
        print("\n" + "="*70)
        print("对战 5: 决策树先手 (X) vs MinMax后手 (O)")
        print("="*70)
        r5 = battle_two_players(tree_player, minmax_player, n_games=args.n_games, verbose=args.verbose)
        print_battle_results(r5, "决策树", "MinMax")
        all_results['tree_vs_minmax'] = r5

        print("\n" + "="*70)
        print("对战 6: MinMax先手 (X) vs 决策树后手 (O)")
        print("="*70)
        r6 = battle_two_players(minmax_player, tree_player, n_games=args.n_games, verbose=args.verbose)
        print_battle_results(r6, "MinMax", "决策树")
        all_results['minmax_vs_tree'] = r6

    # 综合分析
    if args.mode in ['both', 'all']:
        print("\n" + "="*70)
        print("综合分析")
        print("="*70)

        # 统计每个玩家的表现
        players_stats = {
            '神经网络': {'wins': 0, 'losses': 0, 'draws': 0, 'illegal': 0},
            '决策树': {'wins': 0, 'losses': 0, 'draws': 0, 'illegal': 0},
            'MinMax': {'wins': 0, 'losses': 0, 'draws': 0, 'illegal': 0}
        }

        # 统计神经网络 vs 决策树
        if 'nn_vs_tree' in all_results:
            r = all_results['nn_vs_tree']
            players_stats['神经网络']['wins'] += r['player1_wins']
            players_stats['神经网络']['losses'] += r['player2_wins']
            players_stats['神经网络']['draws'] += r['draws']
            players_stats['神经网络']['illegal'] += r['player1_illegal']
            players_stats['决策树']['wins'] += r['player2_wins']
            players_stats['决策树']['losses'] += r['player1_wins']
            players_stats['决策树']['draws'] += r['draws']
            players_stats['决策树']['illegal'] += r['player2_illegal']

        if 'tree_vs_nn' in all_results:
            r = all_results['tree_vs_nn']
            players_stats['决策树']['wins'] += r['player1_wins']
            players_stats['决策树']['losses'] += r['player2_wins']
            players_stats['决策树']['draws'] += r['draws']
            players_stats['决策树']['illegal'] += r['player1_illegal']
            players_stats['神经网络']['wins'] += r['player2_wins']
            players_stats['神经网络']['losses'] += r['player1_wins']
            players_stats['神经网络']['draws'] += r['draws']
            players_stats['神经网络']['illegal'] += r['player2_illegal']

        # 统计神经网络 vs MinMax
        if 'nn_vs_minmax' in all_results:
            r = all_results['nn_vs_minmax']
            players_stats['神经网络']['wins'] += r['player1_wins']
            players_stats['神经网络']['losses'] += r['player2_wins']
            players_stats['神经网络']['draws'] += r['draws']
            players_stats['神经网络']['illegal'] += r['player1_illegal']
            players_stats['MinMax']['wins'] += r['player2_wins']
            players_stats['MinMax']['losses'] += r['player1_wins']
            players_stats['MinMax']['draws'] += r['draws']
            players_stats['MinMax']['illegal'] += r['player2_illegal']

        if 'minmax_vs_nn' in all_results:
            r = all_results['minmax_vs_nn']
            players_stats['MinMax']['wins'] += r['player1_wins']
            players_stats['MinMax']['losses'] += r['player2_wins']
            players_stats['MinMax']['draws'] += r['draws']
            players_stats['MinMax']['illegal'] += r['player1_illegal']
            players_stats['神经网络']['wins'] += r['player2_wins']
            players_stats['神经网络']['losses'] += r['player1_wins']
            players_stats['神经网络']['draws'] += r['draws']
            players_stats['神经网络']['illegal'] += r['player2_illegal']

        # 统计决策树 vs MinMax
        if 'tree_vs_minmax' in all_results:
            r = all_results['tree_vs_minmax']
            players_stats['决策树']['wins'] += r['player1_wins']
            players_stats['决策树']['losses'] += r['player2_wins']
            players_stats['决策树']['draws'] += r['draws']
            players_stats['决策树']['illegal'] += r['player1_illegal']
            players_stats['MinMax']['wins'] += r['player2_wins']
            players_stats['MinMax']['losses'] += r['player1_wins']
            players_stats['MinMax']['draws'] += r['draws']
            players_stats['MinMax']['illegal'] += r['player2_illegal']

        if 'minmax_vs_tree' in all_results:
            r = all_results['minmax_vs_tree']
            players_stats['MinMax']['wins'] += r['player1_wins']
            players_stats['MinMax']['losses'] += r['player2_wins']
            players_stats['MinMax']['draws'] += r['draws']
            players_stats['MinMax']['illegal'] += r['player1_illegal']
            players_stats['决策树']['wins'] += r['player2_wins']
            players_stats['决策树']['losses'] += r['player1_wins']
            players_stats['决策树']['draws'] += r['draws']
            players_stats['决策树']['illegal'] += r['player2_illegal']

        # 打印综合统计
        print("\n总体表现排名:")
        print("-" * 70)
        print(f"{'玩家':<15} {'胜':<8} {'负':<8} {'平':<8} {'胜率':<10} {'非法':<6}")
        print("-" * 70)

        for player_name in ['神经网络', '决策树', 'MinMax']:
            stats = players_stats[player_name]
            total_games = stats['wins'] + stats['losses'] + stats['draws']
            if total_games > 0:
                win_rate = stats['wins'] / total_games * 100
                print(f"{player_name:<15} {stats['wins']:<8} {stats['losses']:<8} "
                      f"{stats['draws']:<8} {win_rate:<9.1f}% {stats['illegal']:<6}")

        # 神经网络 vs 决策树一致性分析
        if 'nn_vs_tree' in all_results and 'tree_vs_nn' in all_results:
            nn_tree_draws = all_results['nn_vs_tree']['draws'] + all_results['tree_vs_nn']['draws']
            nn_tree_total = all_results['nn_vs_tree']['total_games'] + all_results['tree_vs_nn']['total_games']
            nn_tree_draw_rate = nn_tree_draws / nn_tree_total * 100

            print(f"\n神经网络 vs 决策树策略一致性:")
            print(f"  平局率: {nn_tree_draw_rate:.1f}%")
            if nn_tree_draw_rate > 80:
                print("  评价: ✓ 高度一致！策略非常相似。")
            elif nn_tree_draw_rate > 60:
                print("  评价: ✓ 较为一致。")
            elif nn_tree_draw_rate > 40:
                print("  评价: △ 中等一致性。")
            else:
                print("  评价: ✗ 策略差异较大。")

        print("="*70)


if __name__ == "__main__":
    main()
