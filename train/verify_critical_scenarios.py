#!/usr/bin/env python3
"""
验证关键场景生成器
"""
import numpy as np
from train_curriculum_learning import CriticalScenarioGenerator


def main():
    print("验证 CriticalScenarioGenerator...")
    print("=" * 70)

    gen = CriticalScenarioGenerator()
    print(f'\n总场景数: {len(gen.scenarios)}')

    # 统计类型
    win_count = sum(1 for s in gen.scenarios if s['type'] == 'win')
    defend_count = sum(1 for s in gen.scenarios if s['type'] == 'defend')
    print(f'获胜场景: {win_count}')
    print(f'防守场景: {defend_count}')

    # 验证几个场景
    print('\n' + '=' * 70)
    print('示例场景 1 (获胜):')
    print('=' * 70)
    win_scenarios = [s for s in gen.scenarios if s['type'] == 'win']
    if win_scenarios:
        board = win_scenarios[0]['board']
        num_x = int(np.sum(board == 1))
        num_o = int(np.sum(board == -1))
        print(f'X(己方)={num_x}, O(对手)={num_o}')
        print(board.reshape(3, 3))

    print('\n' + '=' * 70)
    print('示例场景 2 (防守):')
    print('=' * 70)
    defend_scenarios = [s for s in gen.scenarios if s['type'] == 'defend']
    if defend_scenarios:
        board = defend_scenarios[0]['board']
        num_x = int(np.sum(board == 1))
        num_o = int(np.sum(board == -1))
        print(f'X(己方)={num_x}, O(对手)={num_o}')
        print(board.reshape(3, 3))

    # 验证场景合法性
    print('\n' + '=' * 70)
    print('合法性检查:')
    print('=' * 70)

    illegal_count = 0
    for i, scenario in enumerate(gen.scenarios[:100]):  # 检查前100个
        board = scenario['board']
        num_x = int(np.sum(board == 1))
        num_o = int(np.sum(board == -1))

        # 检查棋子数量合法性
        if num_x < num_o - 1 or num_x > num_o + 1:
            print(f'✗ 场景 {i}: 棋子数量非法 (X={num_x}, O={num_o})')
            illegal_count += 1

    if illegal_count == 0:
        print('✓ 前100个场景都合法')
    else:
        print(f'✗ 发现 {illegal_count} 个非法场景')


if __name__ == "__main__":
    main()
