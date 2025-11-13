#!/usr/bin/env python3
"""
决策树规则的战术分析
通过枚举所有有效的棋盘状态来分析每条规则的winning/losing条件
"""

import json
import argparse
import numpy as np
from typing import List, Tuple, Dict, Set
from itertools import product


def position_name(pos):
    """位置名称"""
    names = [
        "TL", "TC", "TR",
        "ML", "C", "MR",
        "BL", "BC", "BR"
    ]
    return names[pos]


# TicTacToe中的8条获胜线
WINNING_LINES = [
    [0, 1, 2],  # 上行
    [3, 4, 5],  # 中行
    [6, 7, 8],  # 下行
    [0, 3, 6],  # 左列
    [1, 4, 7],  # 中列
    [2, 5, 8],  # 右列
    [0, 4, 8],  # 对角线 \
    [2, 4, 6],  # 对角线 /
]


def check_winning_condition(board: np.ndarray, player: int) -> List[int]:
    """
    检查哪些获胜线被player威胁（差一步就能赢）
    返回线索引列表，其中player有2个棋子且有1个空位
    """
    threatened_lines = []
    for line_idx, line in enumerate(WINNING_LINES):
        positions = [board[pos] for pos in line]
        player_count = sum(1 for p in positions if p == player)
        empty_count = sum(1 for p in positions if p == 0)

        # 获胜条件：2个己方棋子 + 1个空位
        if player_count == 2 and empty_count == 1:
            threatened_lines.append(line_idx)

    return threatened_lines


def check_losing_condition(board: np.ndarray, player: int) -> List[int]:
    """
    检查哪些获胜线被对手威胁
    返回线索引列表，其中opponent有2个棋子且有1个空位
    """
    opponent = -player
    return check_winning_condition(board, opponent)


def get_winning_move(board: np.ndarray, player: int) -> List[int]:
    """获取player可以立即获胜的位置"""
    winning_moves = []
    for line in WINNING_LINES:
        positions = [board[pos] for pos in line]
        player_count = sum(1 for p in positions if p == player)
        empty_count = sum(1 for p in positions if p == 0)

        if player_count == 2 and empty_count == 1:
            # 找到空位
            for pos in line:
                if board[pos] == 0:
                    winning_moves.append(pos)

    return winning_moves


def get_blocking_move(board: np.ndarray, player: int) -> List[int]:
    """获取player必须阻止对手获胜的位置"""
    opponent = -player
    return get_winning_move(board, opponent)


def parse_constraints_from_antecedents(antecedents: List[List]) -> Dict[int, str]:
    """
    从规则前提条件解析每个位置的约束
    返回字典: 位置 -> 约束类型
    约束类型: 'X', 'O', '.', '?/X', '?/O', '?'
    """
    position_constraints = {i: [] for i in range(9)}

    # 收集所有约束
    for condition in antecedents:
        pos, op, threshold = condition
        position_constraints[pos].append((op, threshold))

    # 分析约束
    result = {}
    for pos in range(9):
        constraints = position_constraints[pos]

        if not constraints:
            result[pos] = '?'
            continue

        # 检查明确的约束
        if any(op == "<=" and th == -0.5 for op, th in constraints):
            result[pos] = 'O'
        elif any(op == ">" and th == 0.5 for op, th in constraints):
            result[pos] = 'X'
        else:
            has_gt_minus_half = any(op == ">" and th == -0.5 for op, th in constraints)
            has_le_half = any(op == "<=" and th == 0.5 for op, th in constraints)

            if has_gt_minus_half and has_le_half:
                result[pos] = '.'
            elif has_gt_minus_half:
                result[pos] = '?/X'  # 不是对手（空或自己）
            elif has_le_half:
                result[pos] = '?/O'  # 不是自己（空或对手）
            else:
                result[pos] = '?'

    return result


def enumerate_valid_boards(constraints: Dict[int, str]) -> List[np.ndarray]:
    """
    枚举给定约束下的所有有效棋盘状态
    约束：X是先手玩家，O是对手
    有效棋盘必须满足：count(X) == count(O)（因为X先手，现在轮到X下棋）
    """
    # 确定每个位置的可能值
    possible_values = {}
    for pos in range(9):
        constraint = constraints[pos]
        if constraint == 'X':
            possible_values[pos] = [1]
        elif constraint == 'O':
            possible_values[pos] = [-1]
        elif constraint == '.':
            possible_values[pos] = [0]
        elif constraint == '?/X':
            possible_values[pos] = [0, 1]  # 空或自己
        elif constraint == '?/O':
            possible_values[pos] = [0, -1]  # 空或对手
        elif constraint == '?':
            possible_values[pos] = [0, 1, -1]  # 任意
        else:
            possible_values[pos] = [0]

    # 生成所有组合
    positions = list(range(9))
    all_combinations = product(*[possible_values[pos] for pos in positions])

    valid_boards = []
    for combination in all_combinations:
        board = np.array(combination, dtype=np.float32)

        x_count = np.sum(board == 1)
        o_count = np.sum(board == -1)

        # X是先手，所以X下棋时：
        # - 数量相等意味着轮到X
        if x_count == o_count:
            valid_boards.append(board)

    return valid_boards


def render_constraint_board(constraints: Dict[int, str]) -> str:
    """渲染约束棋盘（显示规则的约束条件）"""
    lines = []
    lines.append("       ┌─────────┬─────────┬─────────┐")
    for row in range(3):
        cells = []
        for col in range(3):
            idx = row * 3 + col
            constraint = constraints.get(idx, '?')
            # 居中显示，最多9个字符
            cells.append(f"{constraint:^9}")
        lines.append("       │" + "│".join(cells) + "│")
        if row < 2:
            lines.append("       ├─────────┼─────────┼─────────┤")
    lines.append("       └─────────┴─────────┴─────────┘")
    return "\n".join(lines)


def analyze_rule_tactical(rule: Dict) -> Dict:
    """
    对规则进行战术分析
    返回战术统计信息
    """
    constraints = parse_constraints_from_antecedents(rule['antecedents'])

    # 枚举所有有效棋盘
    valid_boards = enumerate_valid_boards(constraints)

    if len(valid_boards) == 0:
        return {
            'num_boards': 0,
            'winning_situations': 0,
            'losing_situations': 0,
            'critical_situations': 0,
            'analysis': '没有找到有效棋盘'
        }

    # 分析每个棋盘
    winning_situations = 0  # 可以获胜的局面数
    losing_situations = 0   # 必须防守的局面数
    critical_situations = 0 # 既能赢又需要防守的局面数

    boards_with_winning_moves = []
    boards_with_losing_threats = []
    boards_with_both = []

    best_action = rule['best_action']

    for board in valid_boards:
        # 检查X（玩家）是否能获胜
        winning_moves = get_winning_move(board, 1)
        # 检查X是否必须阻止O
        blocking_moves = get_blocking_move(board, 1)

        if len(winning_moves) > 0:
            winning_situations += 1
            boards_with_winning_moves.append((board.copy(), winning_moves))

        if len(blocking_moves) > 0:
            losing_situations += 1
            boards_with_losing_threats.append((board.copy(), blocking_moves))

        if len(winning_moves) > 0 and len(blocking_moves) > 0:
            critical_situations += 1
            boards_with_both.append((board.copy(), winning_moves, blocking_moves))

    # 评估best_action是否合适
    correct_wins = 0    # 正确选择获胜动作的次数
    correct_blocks = 0  # 正确选择阻止动作的次数

    for board, winning_moves in boards_with_winning_moves:
        if best_action in winning_moves:
            correct_wins += 1

    for board, blocking_moves in boards_with_losing_threats:
        if best_action in blocking_moves:
            correct_blocks += 1

    return {
        'num_boards': len(valid_boards),
        'winning_situations': winning_situations,
        'losing_situations': losing_situations,
        'critical_situations': critical_situations,
        'correct_wins': correct_wins,
        'correct_blocks': correct_blocks,
        'win_rate': correct_wins / winning_situations if winning_situations > 0 else 0,
        'block_rate': correct_blocks / losing_situations if losing_situations > 0 else 0,
        'sample_boards': {
            'with_winning_moves': boards_with_winning_moves[:3],
            'with_losing_threats': boards_with_losing_threats[:3],
            'critical': boards_with_both[:3]
        }
    }


def format_board_inline(board: np.ndarray) -> str:
    """格式化棋盘为单行字符串（用于简短显示）"""
    symbols = {0: '.', 1: 'X', -1: 'O'}
    result = ""
    for i in range(9):
        result += symbols[board[i]]
    return result


def render_board_simple(board: np.ndarray) -> str:
    """渲染棋盘为二维格式（用于详细显示）"""
    symbols = {0: '.', 1: 'X', -1: 'O'}
    lines = []
    lines.append("       ┌───┬───┬───┐")
    for row in range(3):
        cells = []
        for col in range(3):
            idx = row * 3 + col
            cells.append(f" {symbols[board[idx]]} ")
        lines.append("       │" + "│".join(cells) + "│")
        if row < 2:
            lines.append("       ├───┼───┼───┤")
    lines.append("       └───┴───┴───┘")
    return "\n".join(lines)


def generate_tactical_report(rules: List[Dict], output_file: str):
    """生成战术分析报告"""

    lines = []
    lines.append("=" * 80)
    lines.append("决策树规则的战术分析报告")
    lines.append("=" * 80)
    lines.append("")
    lines.append("本报告分析每条规则的winning/losing条件。")
    lines.append("对于每条规则，我们枚举所有有效的棋盘状态并检查：")
    lines.append("  - Winning情况：玩家(X)可以一步获胜")
    lines.append("  - Losing情况：玩家必须阻止对手(O)获胜")
    lines.append("  - Critical情况：同时存在winning和losing条件")
    lines.append("")
    lines.append("注意：我们只枚举count(X) == count(O)的棋盘，")
    lines.append("      因为X是先手，我们正在决定X的下一步。")
    lines.append("")

    # 汇总统计
    total_rules = len(rules)
    total_analyzed = 0
    rules_with_wins = 0
    rules_with_losses = 0
    rules_with_critical = 0

    print("正在分析规则...")

    detailed_results = []
    for i, rule in enumerate(rules):
        print(f"  分析规则 {i+1}/{len(rules)}...", end='\r')

        analysis = analyze_rule_tactical(rule)

        if analysis['num_boards'] > 0:
            total_analyzed += 1

            if analysis['winning_situations'] > 0:
                rules_with_wins += 1
            if analysis['losing_situations'] > 0:
                rules_with_losses += 1
            if analysis['critical_situations'] > 0:
                rules_with_critical += 1

        detailed_results.append({
            'rule': rule,
            'analysis': analysis
        })

    print()

    # 汇总
    lines.append("=" * 80)
    lines.append("汇总统计")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"总规则数:                    {total_rules}")
    lines.append(f"已分析规则数:                {total_analyzed}")
    lines.append(f"有Winning情况的规则:         {rules_with_wins} ({rules_with_wins/total_analyzed*100:.1f}%)")
    lines.append(f"有Losing情况的规则:          {rules_with_losses} ({rules_with_losses/total_analyzed*100:.1f}%)")
    lines.append(f"有Critical情况的规则:        {rules_with_critical} ({rules_with_critical/total_analyzed*100:.1f}%)")
    lines.append("")

    # 详细分析
    lines.append("=" * 80)
    lines.append("详细规则分析")
    lines.append("=" * 80)

    for i, result in enumerate(detailed_results):
        rule = result['rule']
        analysis = result['analysis']

        if analysis['num_boards'] == 0:
            continue

        lines.append("")
        lines.append("-" * 80)
        lines.append(f"规则 #{i+1}")
        lines.append("-" * 80)
        lines.append(f"优先级:         {rule['priority']:.4f}")
        lines.append(f"支持样本数:     {rule['support_count']} 个")
        lines.append(f"最佳动作:       {rule['best_action']} ({position_name(rule['best_action'])})")
        lines.append("")

        # 显示规则约束对应的棋盘
        constraints = parse_constraints_from_antecedents(rule['antecedents'])
        lines.append("规则约束棋盘 (X=自己, O=对手, .=空, ?=未约束, ?/X=非对手, ?/O=非自己):")
        lines.append(render_constraint_board(constraints))
        lines.append("")

        lines.append(f"有效棋盘数:     {analysis['num_boards']}")
        lines.append(f"Winning局面:    {analysis['winning_situations']} ({analysis['winning_situations']/analysis['num_boards']*100:.1f}%)")
        lines.append(f"Losing局面:     {analysis['losing_situations']} ({analysis['losing_situations']/analysis['num_boards']*100:.1f}%)")
        lines.append(f"Critical局面:   {analysis['critical_situations']} ({analysis['critical_situations']/analysis['num_boards']*100:.1f}%)")
        lines.append("")

        if analysis['winning_situations'] > 0:
            lines.append(f"正确获胜次数:   {analysis['correct_wins']}/{analysis['winning_situations']} ({analysis['win_rate']*100:.1f}%)")
        if analysis['losing_situations'] > 0:
            lines.append(f"正确防守次数:   {analysis['correct_blocks']}/{analysis['losing_situations']} ({analysis['block_rate']*100:.1f}%)")
        lines.append("")

        # 显示示例棋盘
        if len(analysis['sample_boards']['with_winning_moves']) > 0:
            lines.append("可以获胜的示例棋盘:")
            for idx, (board, winning_moves) in enumerate(analysis['sample_boards']['with_winning_moves'], 1):
                lines.append(f"  示例 {idx}: ({format_board_inline(board)}) 可在以下位置获胜: {winning_moves}")
                lines.append(render_board_simple(board))
                lines.append("")

        if len(analysis['sample_boards']['with_losing_threats']) > 0:
            lines.append("必须防守的示例棋盘:")
            for idx, (board, blocking_moves) in enumerate(analysis['sample_boards']['with_losing_threats'], 1):
                lines.append(f"  示例 {idx}: ({format_board_inline(board)}) 必须在以下位置防守: {blocking_moves}")
                lines.append(render_board_simple(board))
                lines.append("")

        if len(analysis['sample_boards']['critical']) > 0:
            lines.append("Critical示例棋盘(既能赢又需要防守):")
            for idx, (board, winning_moves, blocking_moves) in enumerate(analysis['sample_boards']['critical'], 1):
                lines.append(f"  示例 {idx}: ({format_board_inline(board)}) 获胜: {winning_moves}, 防守: {blocking_moves}")
                lines.append(render_board_simple(board))
                lines.append("")

    # 页脚
    lines.append("")
    lines.append("=" * 80)
    lines.append("战术分析结束")
    lines.append("=" * 80)

    # 写入文件
    report_text = "\n".join(lines)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"\n✓ 战术分析已保存到: {output_file}")
    print(f"  已分析规则总数: {total_analyzed}")
    print(f"  有winning情况的规则: {rules_with_wins}")
    print(f"  有losing情况的规则: {rules_with_losses}")


def main():
    parser = argparse.ArgumentParser(description="决策树规则的战术分析")
    parser.add_argument("json_file", type=str, nargs='?',
                       default="log/viper_mask_ppo_tictactoe/viper_mask_ppo_tree_20251107_001324_iter10_samples50000_depth10_leaves50.rules.json",
                       help="规则JSON文件路径")
    parser.add_argument("--output", "-o", type=str, default="tactical_analysis_report.txt",
                       help="输出文件名")
    parser.add_argument("--max-rules", type=int, default=None,
                       help="分析的最大规则数")

    args = parser.parse_args()

    # 加载规则
    print(f"从以下位置加载规则: {args.json_file}")
    with open(args.json_file, 'r') as f:
        data = json.load(f)

    rules = data['rules']
    print(f"已加载 {len(rules)} 条规则")

    if args.max_rules:
        rules = rules[:args.max_rules]
        print(f"限制为前 {args.max_rules} 条规则")

    # 生成战术分析
    generate_tactical_report(rules, args.output)


if __name__ == "__main__":
    main()
