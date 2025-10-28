"""
测试神经网络作为先手和后手的表现
包括：vs Random, vs MinMax, 自己 vs 自己
"""

import gymnasium as gym
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gym_env


def mask_fn(env):
    """返回 action mask"""
    if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'board'):
        board = env.unwrapped.board
    elif hasattr(env, 'board'):
        board = env.board
    elif hasattr(env, 'env') and hasattr(env.env, 'board'):
        board = env.env.board
    else:
        return np.ones(9, dtype=np.int8)

    mask = (board == 0).astype(np.int8)
    return mask


class TicTacToeFirstSecondTester:
    """测试先手和后手表现"""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None

        # 获胜组合
        self.win_combinations = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # 行
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # 列
            [0, 4, 8], [2, 4, 6]              # 对角线
        ]

    def load_model(self):
        """加载模型"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")

        # 创建临时环境用于加载
        temp_env = gym.make('TicTacToe-v0', opponent_type='random')
        temp_env = Monitor(temp_env)
        temp_env = ActionMasker(temp_env, mask_fn)

        self.model = MaskablePPO.load(self.model_path, env=temp_env)
        temp_env.close()

        print(f"✓ 模型加载成功: {self.model_path}\n")

    def test_as_first_player(self, opponent_type='random', num_games=100):
        """测试作为先手(X)的表现"""
        print(f"测试先手(X) vs {opponent_type.upper()} ({num_games} 局)...")

        env = gym.make('TicTacToe-v0', opponent_type=opponent_type)
        env = Monitor(env)
        env = ActionMasker(env, mask_fn)

        wins, losses, draws, illegal = 0, 0, 0, 0
        total_steps = []

        for _ in range(num_games):
            obs, _ = env.reset()
            done = False
            step_count = 0

            while not done:
                mask = mask_fn(env)
                action, _ = self.model.predict(obs, deterministic=True, action_masks=mask)

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                step_count += 1

                if done:
                    total_steps.append(step_count)
                    if 'illegal_move' in info and info['illegal_move']:
                        illegal += 1
                        losses += 1
                    elif reward > 0:
                        wins += 1
                    elif reward < 0:
                        losses += 1
                    else:
                        draws += 1

        env.close()

        return {
            'wins': wins,
            'losses': losses,
            'draws': draws,
            'illegal': illegal,
            'avg_steps': np.mean(total_steps),
            'win_rate': wins / num_games * 100,
            'loss_rate': losses / num_games * 100,
            'draw_rate': draws / num_games * 100
        }

    def test_as_second_player(self, opponent_type='random', num_games=100):
        """
        测试作为后手(O)的表现

        关键：后手看到的局面需要翻转
        - 实际棋盘：X=1, O=-1
        - 后手视角：自己(O)=1, 对手(X)=-1 (翻转符号)
        """
        print(f"测试后手(O) vs {opponent_type.upper()} ({num_games} 局)...")

        from gym_env.policies import RandomPlayerPolicy, MinMaxPlayerPolicy

        # 创建对手策略
        obs_space = gym.spaces.Box(low=-1, high=1, shape=(9,), dtype=np.float32)
        act_space = gym.spaces.Discrete(9)

        if opponent_type == 'random':
            opponent = RandomPlayerPolicy(obs_space, act_space)
        elif opponent_type == 'minmax':
            opponent = MinMaxPlayerPolicy(obs_space, act_space)
        else:
            raise ValueError(f"Unknown opponent type: {opponent_type}")

        wins, losses, draws, illegal = 0, 0, 0, 0
        total_steps = []

        for _ in range(num_games):
            board = np.zeros(9, dtype=np.float32)
            done = False
            step_count = 0

            # 对手(X)先走
            # 对手视角：自己=1 (实际就是X=1)
            opponent_obs = board.copy()
            opponent_action, _ = opponent.predict(opponent_obs, deterministic=False)
            board[opponent_action] = 1  # X

            while not done:
                step_count += 1

                # ⭐ 关键：我方(O)的视角需要翻转
                # 实际棋盘: X=1, O=-1
                # 翻转后: 自己(O)=1, 对手(X)=-1
                my_obs = -board.copy()
                my_mask = (board == 0).astype(np.int8)

                # 我方行动
                my_action, _ = self.model.predict(my_obs, deterministic=True, action_masks=my_mask)

                # 检查合法性
                if board[my_action] != 0:
                    illegal += 1
                    losses += 1
                    done = True
                    break

                board[my_action] = -1  # O

                # 检查我方是否获胜
                if self._check_winner(board, -1):
                    wins += 1
                    done = True
                    total_steps.append(step_count)
                    break

                # 检查平局
                if not np.any(board == 0):
                    draws += 1
                    done = True
                    total_steps.append(step_count)
                    break

                # 对手行动
                # 对手视角：自己=1
                opponent_obs = board.copy()
                opponent_action, _ = opponent.predict(opponent_obs, deterministic=False)
                board[opponent_action] = 1  # X

                # 检查对手是否获胜
                if self._check_winner(board, 1):
                    losses += 1
                    done = True
                    total_steps.append(step_count)
                    break

                # 再次检查平局
                if not np.any(board == 0):
                    draws += 1
                    done = True
                    total_steps.append(step_count)
                    break

        return {
            'wins': wins,
            'losses': losses,
            'draws': draws,
            'illegal': illegal,
            'avg_steps': np.mean(total_steps) if total_steps else 0,
            'win_rate': wins / num_games * 100,
            'loss_rate': losses / num_games * 100,
            'draw_rate': draws / num_games * 100
        }

    def test_self_play(self, num_games=100):
        """
        测试自己和自己下

        关键：
        - 先手(X)视角：自己=1, 对手=-1
        - 后手(O)视角：自己=1 (需要翻转), 对手=-1
        """
        print(f"测试自己 vs 自己 ({num_games} 局)...")

        x_wins, o_wins, draws, illegal = 0, 0, 0, 0
        total_steps = []

        for _ in range(num_games):
            board = np.zeros(9, dtype=np.float32)
            done = False
            step_count = 0

            while not done:
                step_count += 1

                # ===== 先手(X)回合 =====
                # X 的视角：自己=1 (X), 对手=-1 (O)
                x_obs = board.copy()
                x_mask = (board == 0).astype(np.int8)

                x_action, _ = self.model.predict(x_obs, deterministic=True, action_masks=x_mask)

                # 检查合法性
                if board[x_action] != 0:
                    illegal += 1
                    o_wins += 1  # X 非法，O 赢
                    done = True
                    break

                board[x_action] = 1  # X

                # 检查 X 是否获胜
                if self._check_winner(board, 1):
                    x_wins += 1
                    done = True
                    total_steps.append(step_count)
                    break

                # 检查平局
                if not np.any(board == 0):
                    draws += 1
                    done = True
                    total_steps.append(step_count)
                    break

                # ===== 后手(O)回合 =====
                # ⭐ 关键：O 的视角需要翻转
                # 实际棋盘: X=1, O=-1
                # 翻转后: 自己(O)=1, 对手(X)=-1
                o_obs = -board.copy()
                o_mask = (board == 0).astype(np.int8)

                o_action, _ = self.model.predict(o_obs, deterministic=True, action_masks=o_mask)

                # 检查合法性
                if board[o_action] != 0:
                    illegal += 1
                    x_wins += 1  # O 非法，X 赢
                    done = True
                    break

                board[o_action] = -1  # O

                # 检查 O 是否获胜
                if self._check_winner(board, -1):
                    o_wins += 1
                    done = True
                    total_steps.append(step_count)
                    break

                # 再次检查平局
                if not np.any(board == 0):
                    draws += 1
                    done = True
                    total_steps.append(step_count)
                    break

        return {
            'x_wins': x_wins,
            'o_wins': o_wins,
            'draws': draws,
            'illegal': illegal,
            'avg_steps': np.mean(total_steps) if total_steps else 0,
            'x_win_rate': x_wins / num_games * 100,
            'o_win_rate': o_wins / num_games * 100,
            'draw_rate': draws / num_games * 100
        }

    def _check_winner(self, board, player):
        """检查玩家是否获胜"""
        for combo in self.win_combinations:
            if all(board[pos] == player for pos in combo):
                return True
        return False

    def run_full_test(self, num_games=100):
        """完整测试：先手、后手、自己vs自己"""
        print("=" * 70)
        print("先手 vs 后手表现对比测试")
        print("=" * 70)
        print(f"模型: {self.model_path}\n")

        self.load_model()

        opponents = ['random', 'minmax']
        results = {}

        # 测试 vs Random 和 vs MinMax
        for opponent in opponents:
            print("\n" + "=" * 70)
            print(f"对手: {opponent.upper()}")
            print("=" * 70)

            # 先手测试
            first_results = self.test_as_first_player(opponent, num_games)
            print(f"\n先手(X)结果:")
            print(f"  胜率: {first_results['win_rate']:.1f}% ({first_results['wins']}/{num_games})")
            print(f"  负率: {first_results['loss_rate']:.1f}% ({first_results['losses']}/{num_games})")
            print(f"  平局率: {first_results['draw_rate']:.1f}% ({first_results['draws']}/{num_games})")
            print(f"  非法移动: {first_results['illegal']}")
            print(f"  平均步数: {first_results['avg_steps']:.1f}")

            # 后手测试
            second_results = self.test_as_second_player(opponent, num_games)
            print(f"\n后手(O)结果:")
            print(f"  胜率: {second_results['win_rate']:.1f}% ({second_results['wins']}/{num_games})")
            print(f"  负率: {second_results['loss_rate']:.1f}% ({second_results['losses']}/{num_games})")
            print(f"  平局率: {second_results['draw_rate']:.1f}% ({second_results['draws']}/{num_games})")
            print(f"  非法移动: {second_results['illegal']}")
            print(f"  平均步数: {second_results['avg_steps']:.1f}")

            # 对比分析
            win_gap = first_results['win_rate'] - second_results['win_rate']
            print(f"\n对比分析:")
            print(f"  先手优势: {win_gap:+.1f}% (胜率差)")

            if opponent == 'random':
                if first_results['win_rate'] >= 95 and second_results['win_rate'] >= 80:
                    grade = "优秀 ✓✓"
                elif first_results['win_rate'] >= 90 and second_results['win_rate'] >= 70:
                    grade = "良好 ✓"
                elif first_results['win_rate'] >= 80 and second_results['win_rate'] >= 60:
                    grade = "及格"
                else:
                    grade = "不及格 ✗"
                print(f"  评级: {grade}")
                print(f"  期望: 先手>95%, 后手>80%")
            else:  # minmax
                first_ok = first_results['draw_rate'] >= 80
                second_ok = second_results['draw_rate'] >= 60  # 后手对MinMax更难
                if first_ok and second_ok:
                    grade = "优秀 ✓✓"
                elif first_ok or second_ok:
                    grade = "良好 ✓"
                else:
                    grade = "不及格 ✗"
                print(f"  评级: {grade}")
                print(f"  期望: 先手平局>80%, 后手平局>60%")

            results[opponent] = {
                'first': first_results,
                'second': second_results
            }

        # 测试自己 vs 自己
        print("\n" + "=" * 70)
        print("自己 vs 自己")
        print("=" * 70)

        self_play_results = self.test_self_play(num_games)
        print(f"\n结果:")
        print(f"  先手(X)胜率: {self_play_results['x_win_rate']:.1f}% ({self_play_results['x_wins']}/{num_games})")
        print(f"  后手(O)胜率: {self_play_results['o_win_rate']:.1f}% ({self_play_results['o_wins']}/{num_games})")
        print(f"  平局率: {self_play_results['draw_rate']:.1f}% ({self_play_results['draws']}/{num_games})")
        print(f"  非法移动: {self_play_results['illegal']}")
        print(f"  平均步数: {self_play_results['avg_steps']:.1f}")

        x_o_gap = abs(self_play_results['x_win_rate'] - self_play_results['o_win_rate'])
        print(f"\n对比分析:")
        print(f"  先后手胜率差: {x_o_gap:.1f}%")

        if self_play_results['draw_rate'] >= 80:
            grade = "优秀 ✓✓ (高平局率，说明策略稳定)"
        elif self_play_results['draw_rate'] >= 60:
            grade = "良好 ✓"
        elif x_o_gap <= 10:
            grade = "及格 (先后手均衡)"
        else:
            grade = "不及格 ✗ (先后手不均衡)"

        print(f"  评级: {grade}")
        print(f"  期望: 平局率>70% 或 先后手胜率差<10%")

        results['self_play'] = self_play_results

        # 综合评估
        print("\n" + "=" * 70)
        print("综合评估")
        print("=" * 70)

        first_avg = np.mean([results[opp]['first']['win_rate'] for opp in opponents])
        second_avg = np.mean([results[opp]['second']['win_rate'] for opp in opponents])

        print(f"\nvs 外部对手的平均胜率:")
        print(f"  先手: {first_avg:.1f}%")
        print(f"  后手: {second_avg:.1f}%")
        print(f"  差距: {abs(first_avg - second_avg):.1f}%")

        if abs(first_avg - second_avg) < 10:
            print(f"\n✓ 先后手表现均衡（差距 <10%）")
        elif abs(first_avg - second_avg) < 20:
            print(f"\n△ 先后手有一定差距（差距 10-20%）")
        else:
            print(f"\n⚠ 先后手差距较大（差距 >20%），建议调整训练")
            print(f"   提示: 检查 --play-as-o-prob 参数（当前可能不是0.5）")

        # Self-play 分析
        print(f"\n自己 vs 自己:")
        if self_play_results['draw_rate'] >= 70:
            print(f"  ✓ 高平局率 ({self_play_results['draw_rate']:.1f}%)，策略成熟且稳定")
        elif x_o_gap <= 10:
            print(f"  ✓ 先后手均衡（胜率差 {x_o_gap:.1f}%）")
        else:
            print(f"  ⚠ 先后手不均衡（X:{self_play_results['x_win_rate']:.1f}% vs O:{self_play_results['o_win_rate']:.1f}%）")
            print(f"     可能原因: 训练时 play_as_o_prob 不是 0.5，导致先后手能力不对称")

        return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description='测试先手和后手表现')
    parser.add_argument('--model', type=str,
                       default='log/oracle_TicTacToe_ppo_balanced.zip',
                       help='模型路径')
    parser.add_argument('--num-games', type=int, default=100,
                       help='每个测试的游戏局数')

    args = parser.parse_args()

    tester = TicTacToeFirstSecondTester(args.model)
    tester.run_full_test(num_games=args.num_games)


if __name__ == "__main__":
    main()
