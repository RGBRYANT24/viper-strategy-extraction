"""
详细对战调试工具 - Detailed Battle Debugging Tool
=================================================

功能特性 (Features):
-----------------
输出完整的对局记录，包括每一步的：
- ✅ 原始棋盘状态（数组和可视化）
- ✅ 当前玩家ID和是否需要视角转换
- ✅ 转换后的输入（如果进行了视角转换）
- ✅ 合法动作列表
- ✅ 模型的预测动作
- ✅ 动作是否合法
- ✅ 执行后的棋盘状态
- ✅ 游戏结束原因（获胜/平局/非法移动）
- ✅ 支持随机探索以测试不同对局场景
- ✅ **非法移动详细分析**（多局时自动显示）
  - 记录所有非法移动的完整信息（局面、输入、输出）
  - 分析是否所有非法移动都发生在相同局面
  - 按局面分组展示非法移动详情
  - 提供智能诊断建议

使用方法 (Usage):
---------------
基本用法：
    python debug/battle_detailed.py --mode both --n-games 1

查看帮助：
    python debug/battle_detailed.py --help

常用命令示例：
    # 场景1: 验证确定性（无随机探索）
    python debug/battle_detailed.py --mode nn-vs-tree --n-games 1 --epsilon 0.0 --seed 42

    # 场景2: 调试非法移动
    python debug/battle_detailed.py --mode nn-vs-tree --n-games 1

    # 场景3: 多局测试（自动显示统计总结）
    python debug/battle_detailed.py --mode both --n-games 10 --epsilon 0.2

    # 场景4: 可复现的随机测试
    python debug/battle_detailed.py --mode both --n-games 5 --epsilon 0.3 --seed 123

    # 场景5: 批量测试（100局）
    python debug/battle_detailed.py --n-games 100 --epsilon 0.1

参数说明 (Arguments):
-------------------
--mode          对战模式: 'nn-vs-tree', 'tree-vs-nn', 'both'
--n-games       对战局数 (默认: 1)
--epsilon       随机探索概率 0.0-1.0 (默认: 0.0)
                0.0 = 完全确定性，用于验证重复性问题
                0.1-0.3 = 轻度随机性
                0.5+ = 高度随机性
--seed          随机种子，用于复现结果
--oracle-path   神经网络模型路径
--viper-path    决策树模型路径

详细文档：
---------
查看 debug/README.md 获取完整文档和使用示例
"""

import sys
import os
# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
from battle_nn_vs_tree import (
    TicTacToeBattleEnv,
    NeuralNetPlayer,
    DecisionTreePlayer,
    MinMaxPlayer
)


def render_board_inline(board):
    """以一行形式渲染棋盘"""
    symbols = {1: 'X', -1: 'O', 0: '.'}
    return ''.join([symbols[int(cell)] for cell in board])


def render_board_visual(board):
    """以3x3形式渲染棋盘"""
    symbols = {1: 'X', -1: 'O', 0: '.'}
    board_2d = board.reshape(3, 3)
    lines = []
    for i in range(3):
        row = " | ".join([symbols[int(cell)] for cell in board_2d[i]])
        lines.append(f"  {row}")
        if i < 2:
            lines.append(" -----------")
    return '\n'.join(lines)


def battle_with_detailed_logging(player1, player2, player1_name, player2_name, n_games=1,
                                epsilon_random=0.0):
    """
    对战并输出详细日志

    Args:
        player1: 玩家1 (X先手)
        player2: 玩家2 (O后手)
        player1_name: 玩家1名称
        player2_name: 玩家2名称
        n_games: 对战局数
        epsilon_random: 随机探索概率 (0.0-1.0)，玩家有此概率随机选择合法动作

    Returns:
        stats: 统计信息字典，包含 illegal_moves_records 列表
    """
    env = TicTacToeBattleEnv()

    # 统计信息
    stats = {
        'player1_wins': 0,
        'player2_wins': 0,
        'draws': 0,
        'player1_illegal': 0,
        'player2_illegal': 0,
        'total_games': n_games,
        'illegal_moves_records': []  # 记录所有非法移动的详细信息
    }

    for game_idx in range(n_games):
        print("\n" + "="*80)
        print(f"游戏 {game_idx + 1}/{n_games}: {player1_name} (X) vs {player2_name} (O)")
        print("="*80)

        obs = env.reset()
        step_count = 0
        max_steps = 9

        print(f"\n初始棋盘:")
        print(render_board_visual(obs))
        print()

        while not env.done and step_count < max_steps:
            step_count += 1
            print(f"\n{'─'*80}")
            print(f"第 {step_count} 步")
            print(f"{'─'*80}")

            # 确定当前玩家
            if env.current_player == 1:
                current_agent = player1
                agent_name = f"{player1_name} (X)"
                current_player_id = 1
            else:
                current_agent = player2
                agent_name = f"{player2_name} (O)"
                current_player_id = -1

            print(f"\n当前玩家: {agent_name} (player_id={current_player_id})")
            print(f"当前棋盘状态: {render_board_inline(obs)}")
            print(f"当前棋盘数组: {obs}")

            # 获取合法动作
            legal_actions = np.where(obs == 0)[0]
            print(f"合法动作: {legal_actions}")

            # 检查是否使用随机探索
            use_random = np.random.random() < epsilon_random

            if use_random:
                # 随机选择合法动作
                action = np.random.choice(legal_actions)
                print(f"🎲 随机探索模式 (epsilon={epsilon_random})")
                print(f"[随机] 从合法动作 {legal_actions} 中随机选择: {action}")
            else:
                # 根据玩家类型准备输入和预测
                if isinstance(current_agent, (DecisionTreePlayer, NeuralNetPlayer)):
                    # 计算转换后的输入
                    if current_player_id == -1:
                        obs_transformed = -obs
                        print(f"视角转换: YES (因为是O玩家)")
                        print(f"转换后输入: {render_board_inline(obs_transformed)}")
                        print(f"转换后数组: {obs_transformed}")
                    else:
                        obs_transformed = obs
                        print(f"视角转换: NO (因为是X玩家)")
                        print(f"输入 = 原始棋盘")

                    # 预测动作
                    action = current_agent.predict(obs, player_id=current_player_id)

                    # 显示玩家类型特定信息
                    if isinstance(current_agent, DecisionTreePlayer):
                        print(f"[决策树] 预测动作: {action}")
                    elif isinstance(current_agent, NeuralNetPlayer):
                        print(f"[神经网络] 预测动作: {action}")
                else:
                    action = current_agent.predict(obs)
                    print(f"[其他玩家] 预测动作: {action}")

            # 检查动作合法性
            is_legal = action in legal_actions
            print(f"动作 {action} 是否合法: {is_legal}")

            if not is_legal:
                print(f"\n⚠️  非法移动！位置 {action} 已被占用或超出范围")
                print(f"   当前位置值: {obs[action] if 0 <= action < 9 else 'N/A'}")

            # 保存当前棋盘状态（用于记录非法移动）
            board_before_illegal_move = obs.copy()

            # 执行动作
            obs, reward, done, info = env.step(action)

            # 显示执行后的棋盘
            print(f"\n执行后棋盘:")
            print(render_board_visual(obs))

            # 检查游戏结果
            if done:
                print(f"\n{'='*80}")
                print("游戏结束!")
                print(f"{'='*80}")

                if 'illegal_move' in info:
                    print(f"❌ {agent_name} 非法移动，输掉游戏！")
                    print(f"   非法动作: {action}")
                    if 0 <= action < 9:
                        print(f"   该位置状态: {obs[action]}")

                    # 记录非法移动的详细信息
                    illegal_record = {
                        'game_idx': game_idx + 1,
                        'player_name': agent_name,
                        'player_id': current_player_id,
                        'board_before_move': board_before_illegal_move.copy(),  # 非法移动前的棋盘
                        'legal_actions': legal_actions.copy(),
                        'predicted_action': int(action),
                        'step': step_count
                    }

                    # 计算转换后的输入（如果有视角转换）
                    if current_player_id == -1:
                        illegal_record['transformed_board'] = -board_before_illegal_move
                        illegal_record['had_perspective_transform'] = True
                    else:
                        illegal_record['transformed_board'] = board_before_illegal_move.copy()
                        illegal_record['had_perspective_transform'] = False

                    stats['illegal_moves_records'].append(illegal_record)

                    # 更新统计
                    if info['player'] == 1:
                        stats['player1_illegal'] += 1
                        stats['player2_wins'] += 1
                    else:
                        stats['player2_illegal'] += 1
                        stats['player1_wins'] += 1

                elif 'winner' in info:
                    winner_symbol = 'X' if info['winner'] == 1 else 'O'
                    print(f"🎉 {agent_name} 获胜！")
                    print(f"   获胜者: {winner_symbol}")

                    # 更新统计
                    if info['winner'] == 1:
                        stats['player1_wins'] += 1
                    else:
                        stats['player2_wins'] += 1

                elif 'draw' in info:
                    print(f"🤝 平局！")
                    stats['draws'] += 1

                print(f"奖励: {reward}")
                break

        if not done:
            print(f"\n⚠️  达到最大步数 {max_steps}，游戏未结束")

    # 返回统计信息
    return stats


def analyze_illegal_moves(illegal_records):
    """
    分析非法移动记录，判断是否都发生在相同的局面

    Args:
        illegal_records: 非法移动记录列表

    Returns:
        dict: 分析结果
    """
    if len(illegal_records) == 0:
        return {'all_same_board': False, 'unique_boards': 0, 'board_groups': {}}

    # 将棋盘转换为可哈希的元组用于比较
    def board_to_key(board):
        return tuple(board.flatten())

    # 按照棋盘状态分组
    board_groups = {}
    for record in illegal_records:
        # 使用原始棋盘（非法移动前的棋盘）作为key
        key = board_to_key(record['board_before_move'])
        if key not in board_groups:
            board_groups[key] = []
        board_groups[key].append(record)

    # 统计
    unique_boards = len(board_groups)
    all_same_board = (unique_boards == 1)

    return {
        'all_same_board': all_same_board,
        'unique_boards': unique_boards,
        'board_groups': board_groups,
        'total_illegal': len(illegal_records)
    }


def print_illegal_moves_analysis(illegal_records):
    """
    打印非法移动的详细分析

    Args:
        illegal_records: 非法移动记录列表
    """
    if len(illegal_records) == 0:
        return

    print("\n" + "█"*80)
    print("非法移动详细分析 ILLEGAL MOVES ANALYSIS")
    print("█"*80)

    analysis = analyze_illegal_moves(illegal_records)

    print(f"\n总计非法移动: {analysis['total_illegal']} 次")
    print(f"不同的局面数: {analysis['unique_boards']}")

    if analysis['all_same_board']:
        print("\n🔍 关键发现：所有非法移动都发生在 **相同的局面** 下！")
        print("   这表明模型在这个特定局面下存在系统性错误。")
    else:
        print(f"\n🔍 非法移动发生在 {analysis['unique_boards']} 个不同的局面")

    # 显示每个局面的详细信息
    print("\n" + "─"*80)
    print("非法移动局面详情:")
    print("─"*80)

    for idx, (board_key, records) in enumerate(analysis['board_groups'].items(), 1):
        first_record = records[0]
        print(f"\n局面 {idx}/{analysis['unique_boards']} (出现 {len(records)} 次):")
        print("─"*40)

        # 显示棋盘
        print("原始棋盘状态:")
        print(render_board_visual(first_record['board_before_move']))

        if first_record['had_perspective_transform']:
            print("\n转换后的输入（模型实际看到的）:")
            print(render_board_visual(first_record['transformed_board']))

        print(f"\n原始棋盘数组: {first_record['board_before_move']}")
        if first_record['had_perspective_transform']:
            print(f"转换后数组: {first_record['transformed_board']}")

        print(f"合法动作: {first_record['legal_actions']}")
        print(f"模型预测: {first_record['predicted_action']}")
        print(f"是否合法: {first_record['predicted_action'] in first_record['legal_actions']}")

        # 列出所有发生此非法移动的游戏
        game_indices = [r['game_idx'] for r in records]
        print(f"\n发生在游戏: {game_indices}")

    # 总结和建议
    print("\n" + "─"*80)
    print("建议:")
    print("─"*80)

    if analysis['all_same_board']:
        print("  ❌ 严重问题：所有非法移动都在同一局面")
        print("  📌 可能原因：")
        print("     1. 模型在训练时未见过此类局面")
        print("     2. 视角转换后的输入有问题")
        print("     3. 模型对特定棋型的处理有bug")
        print("  💡 建议：")
        print("     1. 检查上述转换后的输入是否符合预期")
        print("     2. 在训练数据中补充此类局面")
        print("     3. 使用 --epsilon 0.3 测试随机探索能否避免")
    else:
        print(f"  ⚠️  非法移动发生在 {analysis['unique_boards']} 个不同局面")
        print("  📌 可能原因：")
        print("     1. 模型整体预测能力不足")
        print("     2. 训练数据不够充分")
        print("  💡 建议：")
        print("     1. 增加训练数据")
        print("     2. 调整模型参数")
        print("     3. 检查训练过程是否正常")

    print("="*80)


def print_battle_summary(stats, player1_name, player2_name):
    """
    打印对战总结

    Args:
        stats: 统计信息字典
        player1_name: 玩家1名称
        player2_name: 玩家2名称
    """
    print("\n" + "█"*80)
    print("对战总结 BATTLE SUMMARY")
    print("█"*80)

    print(f"\n对阵: {player1_name} (X 先手) vs {player2_name} (O 后手)")
    print(f"总局数: {stats['total_games']}")
    print()

    # 胜负统计
    print("─" * 80)
    print("胜负统计:")
    print("─" * 80)
    p1_win_rate = stats['player1_wins'] / stats['total_games'] * 100 if stats['total_games'] > 0 else 0
    p2_win_rate = stats['player2_wins'] / stats['total_games'] * 100 if stats['total_games'] > 0 else 0
    draw_rate = stats['draws'] / stats['total_games'] * 100 if stats['total_games'] > 0 else 0

    print(f"  {player1_name} (X) 获胜: {stats['player1_wins']:3d} 局  ({p1_win_rate:5.1f}%)")
    print(f"  {player2_name} (O) 获胜: {stats['player2_wins']:3d} 局  ({p2_win_rate:5.1f}%)")
    print(f"  平局:                {stats['draws']:3d} 局  ({draw_rate:5.1f}%)")

    # 非法移动统计
    total_illegal = stats['player1_illegal'] + stats['player2_illegal']
    print()
    print("─" * 80)
    print("非法移动统计:")
    print("─" * 80)
    print(f"  总计: {total_illegal} 次")
    print(f"  {player1_name} 非法移动: {stats['player1_illegal']:3d} 次")
    print(f"  {player2_name} 非法移动: {stats['player2_illegal']:3d} 次")

    # 分析和建议
    print()
    print("─" * 80)
    print("分析:")
    print("─" * 80)

    if total_illegal > 0:
        illegal_rate = total_illegal / stats['total_games'] * 100
        print(f"  ⚠️  非法移动率: {illegal_rate:.1f}% ({total_illegal}/{stats['total_games']})")

        if stats['player1_illegal'] > 0:
            print(f"  ⚠️  {player1_name} 存在非法移动问题")
        if stats['player2_illegal'] > 0:
            print(f"  ⚠️  {player2_name} 存在非法移动问题")

        if illegal_rate >= 50:
            print("  ❌ 严重问题：超过50%的对局出现非法移动")
            print("     建议：检查视角转换逻辑和模型训练数据")
        elif illegal_rate >= 10:
            print("  ⚠️  中等问题：超过10%的对局出现非法移动")
            print("     建议：使用 --epsilon 参数测试随机探索能否避免")
    else:
        print("  ✅ 无非法移动，对战过程合法")

    if draw_rate > 80:
        print("  ✅ 高平局率：两个模型策略接近，都接近最优")
    elif draw_rate > 50:
        print("  ✓  中等平局率：模型表现良好")
    elif p1_win_rate > 80 or p2_win_rate > 80:
        winner_name = player1_name if p1_win_rate > 80 else player2_name
        print(f"  ⚠️  {winner_name} 压倒性优势，可能存在问题")

    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="详细对战调试")

    parser.add_argument("--oracle-path", type=str,
                        default="log/oracle_TicTacToe-v0.zip",
                        help="神经网络模型路径")
    parser.add_argument("--viper-path", type=str,
                        default="log/viper_TicTacToe-v0_all-leaves_10.joblib",
                        help="决策树模型路径")
    parser.add_argument("--n-games", type=int, default=1,
                        help="对战局数")
    parser.add_argument("--mode", type=str, default="nn-vs-tree",
                        choices=['nn-vs-tree', 'tree-vs-nn', 'both'],
                        help="对战模式")
    parser.add_argument("--epsilon", type=float, default=0.0,
                        help="随机探索概率 (0.0-1.0)，有此概率随机选择合法动作")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子，用于复现结果")

    args = parser.parse_args()

    # 设置随机种子
    if args.seed is not None:
        np.random.seed(args.seed)
        print(f"已设置随机种子: {args.seed}")

    print("\n" + "="*80)
    print("详细对战调试工具")
    print("="*80)
    if args.epsilon > 0:
        print(f"随机探索概率: {args.epsilon * 100}%")

    # 加载模型（启用调试模式）
    print("\n加载模型...")
    nn_player = NeuralNetPlayer(args.oracle_path, debug=True)
    tree_player = DecisionTreePlayer(args.viper_path, debug=True)

    # 执行对战
    if args.mode in ['nn-vs-tree', 'both']:
        print("\n" + "█"*80)
        print("场景 1: 神经网络先手 (X) vs 决策树后手 (O)")
        print("█"*80)
        stats1 = battle_with_detailed_logging(
            nn_player, tree_player,
            "神经网络", "决策树",
            n_games=args.n_games,
            epsilon_random=args.epsilon
        )

        # 打印总结
        if args.n_games > 1:
            print_battle_summary(stats1, "神经网络", "决策树")

        # 打印非法移动分析
        if len(stats1['illegal_moves_records']) > 0:
            print_illegal_moves_analysis(stats1['illegal_moves_records'])

    if args.mode in ['tree-vs-nn', 'both']:
        # 重置计数器
        nn_player.predict_count = 0
        tree_player.predict_count = 0

        print("\n" + "█"*80)
        print("场景 2: 决策树先手 (X) vs 神经网络后手 (O)")
        print("█"*80)
        stats2 = battle_with_detailed_logging(
            tree_player, nn_player,
            "决策树", "神经网络",
            n_games=args.n_games,
            epsilon_random=args.epsilon
        )

        # 打印总结
        if args.n_games > 1:
            print_battle_summary(stats2, "决策树", "神经网络")

        # 打印非法移动分析
        if len(stats2['illegal_moves_records']) > 0:
            print_illegal_moves_analysis(stats2['illegal_moves_records'])


if __name__ == "__main__":
    main()
