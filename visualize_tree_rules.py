#!/usr/bin/env python3
"""
可视化决策树规则
- 显示棋盘状态
- 显示动作优先级和分布
- 检查策略正常性
"""

import json
import argparse
import numpy as np
from typing import List, Tuple, Dict


def position_name(pos):
    """位置名称"""
    names = [
        "左上", "上中", "右上",
        "左中", "中心", "右中",
        "左下", "下中", "右下"
    ]
    return names[pos]


def parse_board_from_antecedents(antecedents: List[List]) -> Tuple[np.ndarray, Dict[int, str]]:
    """从规则的前提条件解析出棋盘状态

    返回:
        board: 棋盘状态数组 (1=自己, -1=对手, 0=空, 2=未知)
        position_constraints: 每个位置的约束说明
    """
    # 初始化：所有位置都是未知的
    board = np.full(9, 2, dtype=np.float32)
    position_constraints = {i: [] for i in range(9)}

    # 收集每个位置的所有约束
    for condition in antecedents:
        pos, op, threshold = condition
        position_constraints[pos].append((op, threshold))

    # 分析每个位置的约束
    for pos in range(9):
        constraints = position_constraints[pos]

        if not constraints:
            # 没有约束的位置，保持未知
            position_constraints[pos] = "?"
            continue

        # 检查是否明确是对手的棋子
        # X[pos] <= -0.5 表示对手
        if any(op == "<=" and th == -0.5 for op, th in constraints):
            board[pos] = -1
            position_constraints[pos] = "O"
            continue

        # 检查是否明确是自己的棋子
        # X[pos] > 0.5 表示自己
        if any(op == ">" and th == 0.5 for op, th in constraints):
            board[pos] = 1
            position_constraints[pos] = "X"
            continue

        # 分析其他约束组合
        has_gt_minus_half = any(op == ">" and th == -0.5 for op, th in constraints)
        has_le_half = any(op == "<=" and th == 0.5 for op, th in constraints)

        # X[pos] > -0.5 AND X[pos] <= 0.5 表示空位
        # 这意味着 -0.5 < X[pos] <= 0.5
        if has_gt_minus_half and has_le_half:
            board[pos] = 0
            position_constraints[pos] = "."
            continue

        # X[pos] > -0.5 表示不是对手（可能是空或自己）
        if has_gt_minus_half and not has_le_half:
            board[pos] = 2  # 未确定，但不是对手
            position_constraints[pos] = ".或X"
            continue

        # X[pos] <= 0.5 表示不是自己（可能是空或对手）
        if has_le_half and not has_gt_minus_half:
            board[pos] = 2  # 未确定，但不是自己
            position_constraints[pos] = ".或O"
            continue

        # 其他情况保持未知
        position_constraints[pos] = "?"

    return board, position_constraints


def render_board(board: np.ndarray, position_constraints: Dict[int, str] = None) -> str:
    """渲染棋盘 - 显示完整的9格状态"""
    lines = []

    # 使用约束信息来显示更准确的状态
    if position_constraints:
        # 使用约束信息显示
        lines.append("┌─────────┬─────────┬─────────┐")
        for row in range(3):
            cells = []
            for col in range(3):
                idx = row * 3 + col
                constraint = position_constraints.get(idx, "?")
                # 居中显示，最多9个字符
                cells.append(f"{constraint:^9}")
            lines.append("│" + "│".join(cells) + "│")
            if row < 2:
                lines.append("├─────────┼─────────┼─────────┤")
        lines.append("└─────────┴─────────┴─────────┘")
    else:
        # 使用简单符号显示
        symbols = {0: '.', 1: 'X', -1: 'O', 2: '?'}
        lines.append("┌───┬───┬───┐")
        for row in range(3):
            cells = []
            for col in range(3):
                idx = row * 3 + col
                cells.append(f" {symbols[board[idx]]} ")
            lines.append("│" + "│".join(cells) + "│")
            if row < 2:
                lines.append("├───┼───┼───┤")
        lines.append("└───┴───┴───┘")

    # 添加位置编号参考
    lines.append("")
    lines.append("位置编号参考:")
    lines.append("┌───┬───┬───┐")
    for row in range(3):
        cells = []
        for col in range(3):
            idx = row * 3 + col
            cells.append(f" {idx} ")
        lines.append("│" + "│".join(cells) + "│")
        if row < 2:
            lines.append("├───┼───┼───┤")
    lines.append("└───┴───┴───┘")

    # 添加图例
    lines.append("")
    lines.append("图例: X=自己的棋子, O=对手的棋子, .=空位, ?=未约束, .或X/.或O=部分约束")

    return "\n".join(lines)


def format_condition(condition: List) -> str:
    """格式化单个条件"""
    pos, op, threshold = condition
    pos_name = position_name(pos)

    if op == "<=" and threshold == -0.5:
        return f"{pos_name}(#{pos}) = O(对手)       [{op} {threshold}]"
    elif op == ">" and threshold == 0.5:
        return f"{pos_name}(#{pos}) = X(自己)       [{op} {threshold}]"
    elif op == "<=" and threshold == 0.5:
        return f"{pos_name}(#{pos}) ≤ 0.5 (非自己, 可能是空或对手) [{op} {threshold}]"
    elif op == ">" and threshold == -0.5:
        return f"{pos_name}(#{pos}) > -0.5 (非对手, 可能是空或自己) [{op} {threshold}]"
    else:
        return f"{pos_name}(#{pos}) {op} {threshold:.2f}"


def get_action_distribution_from_output(output_vector: List[float]) -> List[Tuple[int, float]]:
    """从输出向量获取动作分布（按优先级排序）"""
    # output_vector是logits，值越大越优先
    actions_with_scores = [(action, score) for action, score in enumerate(output_vector)]
    # 按分数从高到低排序
    actions_with_scores.sort(key=lambda x: x[1], reverse=True)
    return actions_with_scores


def softmax(logits: List[float]) -> np.ndarray:
    """计算softmax概率分布"""
    logits = np.array(logits)
    exp_logits = np.exp(logits - np.max(logits))  # 数值稳定性
    return exp_logits / exp_logits.sum()


def check_rule_sanity(rule: Dict, board: np.ndarray, position_constraints: Dict[int, str]) -> List[str]:
    """检查规则的合理性"""
    issues = []

    best_action = rule['best_action']
    output_vector = rule['output_vector']

    # 检查1: 最佳动作的位置应该是空的或未确定的
    if board[best_action] == -1:
        issues.append(f"❌ 最佳动作{best_action}({position_name(best_action)})位置已被对手占据！")
    elif board[best_action] == 1:
        issues.append(f"❌ 最佳动作{best_action}({position_name(best_action)})位置已被自己占据！")
    elif board[best_action] == 2:
        constraint = position_constraints.get(best_action, "未知")
        issues.append(f"⚠️ 最佳动作{best_action}({position_name(best_action)})位置未完全确定 ({constraint})")

    # 检查2: 被明确占据的位置不应该有高优先级
    actions_with_scores = get_action_distribution_from_output(output_vector)
    for rank, (action, score) in enumerate(actions_with_scores[:3], 1):  # 检查前3个优先级
        if board[action] == -1:
            issues.append(f"⚠️ 动作{action}({position_name(action)})排名第{rank}但已被对手占据 (score={score:.2f})")
        elif board[action] == 1:
            issues.append(f"⚠️ 动作{action}({position_name(action)})排名第{rank}但已被自己占据 (score={score:.2f})")

    # 检查3: 输出向量中的极端值
    max_score = max(output_vector)
    min_score = min(output_vector)
    if max_score - min_score < 0.5:
        issues.append(f"⚠️ 动作分数范围很小({max_score - min_score:.2f})，策略可能不够确定")

    return issues


def visualize_rule(rule: Dict, rule_idx: int, verbose: bool = True):
    """可视化单个规则"""
    print("\n" + "=" * 80)
    print(f"规则 #{rule_idx + 1}")
    print("=" * 80)

    # 基本信息
    print(f"优先级: {rule['priority']:.2f}")
    print(f"支持样本数: {rule['support_count']}")
    print(f"最佳动作: {rule['best_action']} ({position_name(rule['best_action'])})")
    print()

    # 显示条件
    print("前提条件:")
    for i, condition in enumerate(rule['antecedents'], 1):
        print(f"  {i}. {format_condition(condition)}")
    print()

    # 解析棋盘状态
    board, position_constraints = parse_board_from_antecedents(rule['antecedents'])
    print("棋盘状态 (根据条件推断, 显示全部9个位置):")
    print(render_board(board, position_constraints))
    print()

    # 动作分布
    output_vector = rule['output_vector']
    actions_with_scores = get_action_distribution_from_output(output_vector)
    probs = softmax(output_vector)

    print("动作优先级排序 (从高到低):")
    print(f"{'排名':<6} {'动作':<6} {'位置':<12} {'分数(logit)':<15} {'概率':<10} {'状态':<10} {'标记'}")
    print("-" * 85)

    for rank, (action, score) in enumerate(actions_with_scores, 1):
        prob = probs[action]
        # 根据board状态显示更详细的信息
        if board[action] == 0:
            status = "空位"
        elif board[action] == 1:
            status = "已占(X)"
        elif board[action] == -1:
            status = "已占(O)"
        else:
            status = "未确定"

        marker = "✓最佳" if action == rule['best_action'] else ""
        print(f"{rank:<6} {action:<6} {position_name(action):<12} {score:<15.3f} {prob:<10.2%} {status:<10} {marker}")
    print()

    # 合理性检查
    issues = check_rule_sanity(rule, board, position_constraints)
    if issues:
        print("❌ 发现潜在问题:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("✅ 规则看起来正常")

    if verbose:
        print("\n原始规则字符串:")
        print(f"  {rule['rule_string']}")


def visualize_rules_summary(rules: List[Dict], top_n: int = 10):
    """显示规则摘要"""
    print("\n" + "=" * 80)
    print(f"规则摘要 (前 {top_n} 条)")
    print("=" * 80)
    print()

    print(f"{'#':<5} {'优先级':<10} {'支持数':<10} {'最佳动作':<15} {'条件数'}")
    print("-" * 80)

    for i, rule in enumerate(rules[:top_n], 1):
        action = rule['best_action']
        print(f"{i:<5} {rule['priority']:<10.2f} {rule['support_count']:<10} "
              f"{action}:{position_name(action):<10} {len(rule['antecedents'])}")


def interactive_mode(rules: List[Dict]):
    """交互模式，让用户选择要查看的规则"""
    while True:
        print("\n" + "=" * 80)
        print("交互模式")
        print("=" * 80)
        print(f"共有 {len(rules)} 条规则")
        print()
        print("命令:")
        print("  1-N    : 查看规则#N")
        print("  summary: 查看摘要")
        print("  all    : 查看所有规则")
        print("  q/quit : 退出")
        print()

        cmd = input("请输入命令: ").strip().lower()

        if cmd in ['q', 'quit', 'exit']:
            break
        elif cmd == 'summary':
            n = input("显示前几条？(默认10): ").strip()
            n = int(n) if n.isdigit() else 10
            visualize_rules_summary(rules, top_n=n)
        elif cmd == 'all':
            for i, rule in enumerate(rules):
                visualize_rule(rule, i, verbose=False)
                input("\n按Enter继续...")
        elif cmd.isdigit():
            idx = int(cmd) - 1
            if 0 <= idx < len(rules):
                visualize_rule(rules[idx], idx, verbose=True)
            else:
                print(f"错误: 规则编号超出范围 (1-{len(rules)})")
        else:
            print("未知命令")


def main():
    parser = argparse.ArgumentParser(description="可视化决策树规则")
    parser.add_argument("json_file", type=str, nargs='?',
                       default="log/viper_mask_ppo_tictactoe/viper_mask_ppo_tree_20251107_001324_iter10_samples50000_depth10_leaves50.rules.json",
                       help="规则JSON文件路径")
    parser.add_argument("--rule", type=int, help="只显示特定规则编号")
    parser.add_argument("--top", type=int, default=10, help="显示前N条规则")
    parser.add_argument("--interactive", "-i", action="store_true", help="交互模式")
    parser.add_argument("--all", action="store_true", help="显示所有规则")

    args = parser.parse_args()

    # 读取JSON文件
    print(f"读取规则文件: {args.json_file}")
    with open(args.json_file, 'r') as f:
        data = json.load(f)

    rules = data['rules']
    print(f"共加载 {len(rules)} 条规则")

    # 显示统计信息
    if 'statistics' in data:
        stats = data['statistics']
        print("\n统计信息:")
        print(f"  规则数: {stats['n_rules']}")
        print(f"  平均条件数: {stats['avg_antecedents']:.2f}")
        print(f"  优先级范围: [{stats['priority_min']:.2f}, {stats['priority_max']:.2f}]")
        print(f"  优先级均值: {stats['priority_mean']:.2f} ± {stats['priority_std']:.2f}")

    # 根据参数决定显示模式
    if args.interactive:
        interactive_mode(rules)
    elif args.rule is not None:
        idx = args.rule - 1
        if 0 <= idx < len(rules):
            visualize_rule(rules[idx], idx, verbose=True)
        else:
            print(f"错误: 规则编号超出范围 (1-{len(rules)})")
    elif args.all:
        for i, rule in enumerate(rules):
            visualize_rule(rule, i, verbose=False)
    else:
        visualize_rules_summary(rules, top_n=args.top)
        print("\n使用 --rule N 查看特定规则详情")
        print("使用 --interactive 进入交互模式")


if __name__ == "__main__":
    main()
