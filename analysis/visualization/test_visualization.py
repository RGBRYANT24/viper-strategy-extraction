#!/usr/bin/env python3
"""
测试可视化工具的条件解析功能
"""

import numpy as np
from visualize_tree_rules import parse_board_from_antecedents, render_board, format_condition


def test_case(name: str, antecedents):
    """测试一个案例"""
    print("\n" + "=" * 80)
    print(f"测试案例: {name}")
    print("=" * 80)

    print("\n前提条件:")
    for i, cond in enumerate(antecedents, 1):
        print(f"  {i}. {format_condition(cond)}")

    board, constraints = parse_board_from_antecedents(antecedents)

    print("\n棋盘解析结果 (显示全部9个位置):")
    print(render_board(board, constraints))

    print("\n每个位置的约束说明:")
    for pos in range(9):
        from visualize_tree_rules import position_name
        print(f"  位置{pos}({position_name(pos)}): {constraints.get(pos, '?')}")


def main():
    print("=" * 80)
    print("可视化工具测试")
    print("=" * 80)

    # 测试案例1: 明确的棋盘状态
    test_case(
        "案例1: 明确的棋盘状态",
        [
            [0, "<=", -0.5],  # 位置0是对手
            [1, ">", 0.5],    # 位置1是自己
            [4, ">", 0.5],    # 位置4(中心)是自己
            [8, "<=", -0.5],  # 位置8是对手
        ]
    )

    # 测试案例2: 包含不确定的状态
    test_case(
        "案例2: 包含不确定的状态",
        [
            [1, "<=", 0.5],   # 位置1不是自己(可能是空或对手)
            [6, ">", -0.5],   # 位置6不是对手(可能是空或自己)
            [6, ">", 0.5],    # 位置6是自己
            [0, "<=", -0.5],  # 位置0是对手
            [3, ">", -0.5],   # 位置3不是对手
            [5, ">", -0.5],   # 位置5不是对手
            [8, "<=", -0.5],  # 位置8是对手
        ]
    )

    # 测试案例3: 组合约束推断空位
    test_case(
        "案例3: 组合约束推断空位",
        [
            [2, ">", -0.5],   # 位置2不是对手
            [2, "<=", 0.5],   # 位置2不是自己 -> 所以是空位
            [4, ">", 0.5],    # 位置4是自己
        ]
    )

    print("\n" + "=" * 80)
    print("测试完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()
