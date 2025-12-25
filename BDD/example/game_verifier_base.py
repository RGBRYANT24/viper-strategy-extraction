"""
通用棋盘游戏 BDD 验证器基类
支持: 井字棋, Connect 4, 五子棋等

编码约定:
  00: Empty (空)
  10: Player (我方)
  01: Opponent (敌方)
  11: Invalid (非法状态)
"""

from dd.autoref import BDD
from abc import ABC, abstractmethod


class GameVerifierBase(ABC):
    """棋盘游戏 BDD 验证器基类"""
    
    def __init__(self, rows: int, cols: int, win_length: int):
        """
        Args:
            rows: 棋盘行数
            cols: 棋盘列数
            win_length: 连成几子算赢 (井字棋=3, Connect4=4, 五子棋=5)
        """
        self.bdd = BDD()
        self.rows = rows
        self.cols = cols
        self.N = rows * cols  # 总格子数
        self.win_length = win_length
        
        # 变量声明
        self.x_vars = []  # 当前状态变量
        self.y_vars = []  # 下一状态变量
        self.prime_map = {}  # x -> y 映射
        self.unprime_map = {}  # y -> x 映射
        
        self._declare_variables()
        
        # 预计算 BDD
        self._valid_state_bdd = None
        self._player_win_bdd = None
        self._opponent_win_bdd = None
        
    def _declare_variables(self):
        """声明 BDD 变量"""
        for i in range(self.N):
            xi_0 = f'x_{i}_0'
            xi_1 = f'x_{i}_1'
            yi_0 = f'y_{i}_0'
            yi_1 = f'y_{i}_1'
            
            self.bdd.declare(xi_0, xi_1, yi_0, yi_1)
            self.x_vars.extend([xi_0, xi_1])
            self.y_vars.extend([yi_0, yi_1])
            
            self.prime_map[xi_0] = yi_0
            self.prime_map[xi_1] = yi_1
            self.unprime_map[yi_0] = xi_0
            self.unprime_map[yi_1] = xi_1

    def _idx(self, row: int, col: int) -> int:
        """(row, col) -> 线性索引"""
        return row * self.cols + col
    
    def _pos(self, idx: int) -> tuple:
        """线性索引 -> (row, col)"""
        return idx // self.cols, idx % self.cols

    def _get_cell_expr(self, idx: int, prefix: str = 'x') -> tuple:
        """获取格子的两个 BDD 变量名"""
        return f'{prefix}_{idx}_0', f'{prefix}_{idx}_1'

    # =========================================================
    # 状态编码
    # =========================================================
    
    def cell_is_empty(self, idx: int, prefix: str = 'x'):
        """格子为空 (00)"""
        v0, v1 = self._get_cell_expr(idx, prefix)
        return self.bdd.add_expr(f'~{v0} & ~{v1}')
    
    def cell_is_player(self, idx: int, prefix: str = 'x'):
        """格子是我方 (10)"""
        v0, v1 = self._get_cell_expr(idx, prefix)
        return self.bdd.add_expr(f'{v0} & ~{v1}')
    
    def cell_is_opponent(self, idx: int, prefix: str = 'x'):
        """格子是敌方 (01)"""
        v0, v1 = self._get_cell_expr(idx, prefix)
        return self.bdd.add_expr(f'~{v0} & {v1}')
    
    def cell_is_valid(self, idx: int, prefix: str = 'x'):
        """格子不是非法状态 (非 11)"""
        v0, v1 = self._get_cell_expr(idx, prefix)
        return self.bdd.add_expr(f'~({v0} & {v1})')

    # =========================================================
    # 胜利条件检测 (终局)
    # =========================================================
    
    def _get_all_lines(self) -> list:
        """获取所有可能形成连线的位置列表"""
        lines = []
        
        # 横向
        for r in range(self.rows):
            for c in range(self.cols - self.win_length + 1):
                line = [self._idx(r, c + i) for i in range(self.win_length)]
                lines.append(line)
        
        # 纵向
        for r in range(self.rows - self.win_length + 1):
            for c in range(self.cols):
                line = [self._idx(r + i, c) for i in range(self.win_length)]
                lines.append(line)
        
        # 主对角线 (左上到右下)
        for r in range(self.rows - self.win_length + 1):
            for c in range(self.cols - self.win_length + 1):
                line = [self._idx(r + i, c + i) for i in range(self.win_length)]
                lines.append(line)
        
        # 副对角线 (右上到左下)
        for r in range(self.rows - self.win_length + 1):
            for c in range(self.win_length - 1, self.cols):
                line = [self._idx(r + i, c - i) for i in range(self.win_length)]
                lines.append(line)
        
        return lines

    def get_player_win_bdd(self, prefix: str = 'x'):
        """构建我方获胜的 BDD (任意一条连线全是 Player)"""
        win_bdd = self.bdd.false
        
        for line in self._get_all_lines():
            line_win = self.bdd.true
            for idx in line:
                line_win = line_win & self.cell_is_player(idx, prefix)
            win_bdd = win_bdd | line_win
        
        return win_bdd
    
    def get_opponent_win_bdd(self, prefix: str = 'x'):
        """构建敌方获胜的 BDD"""
        win_bdd = self.bdd.false
        
        for line in self._get_all_lines():
            line_win = self.bdd.true
            for idx in line:
                line_win = line_win & self.cell_is_opponent(idx, prefix)
            win_bdd = win_bdd | line_win
        
        return win_bdd
    
    def get_draw_bdd(self, prefix: str = 'x'):
        """构建平局的 BDD (棋盘满且无人获胜)"""
        # 所有格子都不为空
        board_full = self.bdd.true
        for i in range(self.N):
            board_full = board_full & ~self.cell_is_empty(i, prefix)
        
        # 且双方都没获胜
        no_winner = ~self.get_player_win_bdd(prefix) & ~self.get_opponent_win_bdd(prefix)
        
        return board_full & no_winner
    
    def is_terminal_state(self, prefix: str = 'x'):
        """判断是否终局 (任意一方获胜或平局)"""
        return self.get_player_win_bdd(prefix) | self.get_opponent_win_bdd(prefix) | self.get_draw_bdd(prefix)

    # =========================================================
    # 合法落子检测 (Mask)
    # =========================================================
    
    def get_valid_actions_bdd(self, prefix: str = 'x'):
        """
        返回所有合法落子位置的 BDD
        默认：空位即可落子
        子类可覆盖（如 Connect4 需要考虑重力）
        """
        # 未终局
        not_terminal = ~self.is_terminal_state(prefix)
        return not_terminal

    def is_valid_action(self, action_idx: int, prefix: str = 'x'):
        """检查特定 action 是否合法（该位置为空）"""
        return self.cell_is_empty(action_idx, prefix) & ~self.is_terminal_state(prefix)

    # =========================================================
    # 规则解析 (从决策树)
    # =========================================================
    
    def _constraint_to_bdd(self, index: int, op: str, threshold: float):
        """将数值约束转换为 BDD"""
        v0, v1 = self._get_cell_expr(index, 'x')
        
        is_player = self.cell_is_player(index, 'x')
        is_opponent = self.cell_is_opponent(index, 'x')
        
        res = self.bdd.false
        
        if op == '>' and threshold >= 0.5:
            res = is_player
        elif op == '<=' and threshold >= 0.5:
            res = ~is_player
        elif op == '<=' and threshold <= -0.5:
            res = is_opponent
        elif op == '>' and threshold >= -0.5:
            res = ~is_opponent
        else:
            print(f"Warning: 未处理的约束条件: X[{index}] {op} {threshold}")
            res = self.bdd.true
        
        valid_state = self.cell_is_valid(index, 'x')
        return res & valid_state

    def parse_rule(self, rule_json: dict):
        """解析决策树规则，返回 (Condition_BDD, Action_Index)"""
        cond_bdd = self.bdd.true
        
        for ant in rule_json['antecedents']:
            idx, op, val = ant
            clause = self._constraint_to_bdd(idx, op, val)
            cond_bdd = cond_bdd & clause
        
        action = rule_json['best_action']
        return cond_bdd, action

    def parse_probability_rule(self, rule_json: dict):
        """
        解析概率输出的决策树规则
        rule_json 格式:
        {
            "antecedents": [...],
            "action_probs": [0.1, 0.0, 0.5, ...],  # 各位置的概率
        }
        返回 (Condition_BDD, Best_Action_Index)
        """
        cond_bdd = self.bdd.true
        
        for ant in rule_json['antecedents']:
            idx, op, val = ant
            clause = self._constraint_to_bdd(idx, op, val)
            cond_bdd = cond_bdd & clause
        
        # 从概率向量中选择最高概率的合法动作
        probs = rule_json.get('action_probs', [])
        if probs:
            # 找最大概率的 action
            best_action = max(range(len(probs)), key=lambda i: probs[i])
        else:
            best_action = rule_json.get('best_action', 0)
        
        return cond_bdd, best_action

    # =========================================================
    # 状态转移
    # =========================================================
    
    def build_move_transition(self, condition_bdd, action_idx: int, is_player: bool = True):
        """构建落子后的状态转移 BDD"""
        # 目标位置变成 Player(10) 或 Opponent(01)
        y0, y1 = self._get_cell_expr(action_idx, 'y')
        if is_player:
            target_change = self.bdd.add_expr(f'{y0} & ~{y1}')
        else:
            target_change = self.bdd.add_expr(f'~{y0} & {y1}')
        
        # 其他位置保持不变
        frame_cond = self.bdd.true
        for i in range(self.N):
            if i == action_idx:
                continue
            x0, x1 = self._get_cell_expr(i, 'x')
            y0, y1 = self._get_cell_expr(i, 'y')
            eq = self.bdd.add_expr(f'({x0} <-> {y0}) & ({x1} <-> {y1})')
            frame_cond = frame_cond & eq
        
        # 确保 action 位置原本为空
        action_valid = self.cell_is_empty(action_idx, 'x')
        
        return condition_bdd & action_valid & target_change & frame_cond

    def get_opponent_transition(self):
        """构建对手所有可能落子的转移关系"""
        t_opp = self.bdd.false
        
        for i in range(self.N):
            move_i = self.build_move_transition(self.bdd.true, i, is_player=False)
            t_opp = t_opp | move_i
        
        return t_opp

    # =========================================================
    # 可视化
    # =========================================================
    
    def print_board(self, state_dict: dict):
        """打印棋盘状态"""
        for r in range(self.rows):
            row_str = ""
            for c in range(self.cols):
                idx = self._idx(r, c)
                b0 = state_dict.get(f'x_{idx}_0', False)
                b1 = state_dict.get(f'x_{idx}_1', False)
                
                if not b0 and not b1:
                    ch = '.'
                elif b0 and not b1:
                    ch = 'X'  # Player
                elif not b0 and b1:
                    ch = 'O'  # Opponent
                else:
                    ch = 'E'  # Error
                
                row_str += f" {ch} "
                if c < self.cols - 1:
                    row_str += "|"
            
            print(row_str)
            if r < self.rows - 1:
                print("-" * (4 * self.cols - 1))

    # =========================================================
    # 验证策略
    # =========================================================
    
    def verify_strategy(self, rules_data: list, max_steps: int = None):
        """验证策略的状态转移"""
        if max_steps is None:
            max_steps = self.N  # 最多下满棋盘
        
        print(f"=== {self.__class__.__name__} 策略验证 ===")
        print(f"棋盘: {self.rows}×{self.cols}, 连{self.win_length}子获胜")
        print(f"BDD 变量数: {len(self.x_vars) + len(self.y_vars)}")
        
        # 构建我方策略
        print("\n构建策略 BDD...")
        strategy_trans = self.bdd.false
        for rule in rules_data:
            cond, action = self.parse_rule(rule)
            rule_trans = self.build_move_transition(cond, action, is_player=True)
            strategy_trans = strategy_trans | rule_trans
        
        print(f"策略 BDD 节点数: {len(strategy_trans)}")
        
        # 构建对手转移
        print("构建对手逻辑 BDD...")
        opp_trans = self.get_opponent_transition()
        print(f"对手 BDD 节点数: {len(opp_trans)}")
        
        # 构建胜利条件
        print("构建终局条件 BDD...")
        player_win = self.get_player_win_bdd('x')
        opponent_win = self.get_opponent_win_bdd('x')
        draw = self.get_draw_bdd('x')
        print(f"我方获胜 BDD 节点数: {len(player_win)}")
        print(f"敌方获胜 BDD 节点数: {len(opponent_win)}")
        
        # 选取初始状态
        print("\n--- 验证演示 ---")
        first_cond, _ = self.parse_rule(rules_data[0])
        start_state = self.bdd.pick(first_cond)
        
        if start_state is None:
            print("错误：无法找到满足规则的初始状态")
            return
        
        print("初始状态:")
        self.print_board(start_state)
        
        current_bdd = self.bdd.cube(start_state)
        
        for step in range(1, max_steps + 1):
            print(f"\n=== Round {step} ===")
            
            # 检查终局
            if (current_bdd & player_win) != self.bdd.false:
                print("🎉 我方获胜!")
                break
            if (current_bdd & opponent_win) != self.bdd.false:
                print("💀 敌方获胜!")
                break
            if (current_bdd & draw) != self.bdd.false:
                print("🤝 平局!")
                break
            
            # 我方回合
            print("[Player Turn]")
            next_y = self.bdd.quantify(current_bdd & strategy_trans, self.x_vars, forall=False)
            
            if next_y == self.bdd.false:
                print("我方无路可走")
                break
            
            my_move = self.bdd.let(self.unprime_map, next_y)
            example = self.bdd.pick(my_move)
            if example:
                self.print_board(example)
            
            # 对手回合
            print("\n[Opponent Turn]")
            opp_y = self.bdd.quantify(my_move & opp_trans, self.x_vars, forall=False)
            
            if opp_y == self.bdd.false:
                print("对手无路可走")
                # 检查是否我方刚获胜
                if (my_move & player_win) != self.bdd.false:
                    print("🎉 我方获胜!")
                break
            
            current_bdd = self.bdd.let(self.unprime_map, opp_y)
            example = self.bdd.pick(current_bdd)
            if example:
                self.print_board(example)
        
        print("\n验证完成")


# =========================================================
# 具体游戏实现
# =========================================================

class TicTacToeVerifier(GameVerifierBase):
    """井字棋验证器"""
    def __init__(self):
        super().__init__(rows=3, cols=3, win_length=3)


class Gomoku9x9Verifier(GameVerifierBase):
    """9×9 五子棋验证器"""
    def __init__(self):
        super().__init__(rows=9, cols=9, win_length=5)


class Connect4Verifier(GameVerifierBase):
    """Connect 4 验证器 (带重力)"""
    def __init__(self):
        super().__init__(rows=6, cols=7, win_length=4)
    
    def is_valid_action(self, col: int, prefix: str = 'x'):
        """Connect4: action 是列号，需要落到该列最底部空位"""
        # 检查该列最顶部是否为空（即该列还没满）
        top_idx = self._idx(0, col)
        return self.cell_is_empty(top_idx, prefix) & ~self.is_terminal_state(prefix)
    
    def build_move_transition(self, condition_bdd, action_col: int, is_player: bool = True):
        """
        Connect4 特殊转移: 棋子落到该列最底部空位
        """
        trans = self.bdd.false
        
        # 对于每一行，检查是否是该列的最底部空位
        for r in range(self.rows):
            idx = self._idx(r, action_col)
            
            # 当前位置为空
            is_empty_here = self.cell_is_empty(idx, 'x')
            
            # 下方所有位置都不为空（或者已经是最底行）
            below_all_filled = self.bdd.true
            for r_below in range(r + 1, self.rows):
                below_idx = self._idx(r_below, action_col)
                below_all_filled = below_all_filled & ~self.cell_is_empty(below_idx, 'x')
            
            # 落子位置变成 Player 或 Opponent
            y0, y1 = self._get_cell_expr(idx, 'y')
            if is_player:
                become = self.bdd.add_expr(f'{y0} & ~{y1}')
            else:
                become = self.bdd.add_expr(f'~{y0} & {y1}')
            
            # 其他位置不变
            frame = self.bdd.true
            for i in range(self.N):
                if i == idx:
                    continue
                x0, x1 = self._get_cell_expr(i, 'x')
                y0, y1 = self._get_cell_expr(i, 'y')
                frame = frame & self.bdd.add_expr(f'({x0} <-> {y0}) & ({x1} <-> {y1})')
            
            row_trans = condition_bdd & is_empty_here & below_all_filled & become & frame
            trans = trans | row_trans
        
        return trans
    
    def get_opponent_transition(self):
        """Connect4: 对手落子也遵循重力规则"""
        t_opp = self.bdd.false
        for col in range(self.cols):
            move = self.build_move_transition(self.bdd.true, col, is_player=False)
            t_opp = t_opp | move
        return t_opp


# =========================================================
# 测试
# =========================================================

if __name__ == "__main__":
    # 测试井字棋
    print("=" * 50)
    print("测试井字棋验证器")
    print("=" * 50)
    
    ttt = TicTacToeVerifier()
    
    test_rule = {
        "rules": [
            {
                "antecedents": [
                    [4, "<=", 0.5],  # 中心不是 Player
                ],
                "best_action": 4  # 下中心
            }
        ]
    }
    
    ttt.verify_strategy(test_rule['rules'], max_steps=5)
    
    print("\n")
    print("=" * 50)
    print("测试 9×9 五子棋验证器 (基础测试)")
    print("=" * 50)
    
    gomoku = Gomoku9x9Verifier()
    print(f"棋盘大小: {gomoku.rows}×{gomoku.cols}")
    print(f"总格子数: {gomoku.N}")
    print(f"BDD 变量数: {len(gomoku.x_vars) + len(gomoku.y_vars)}")
    print(f"连线数 (胜利条件): {len(gomoku._get_all_lines())}")
    
    # 简单测试：构建单个格子空位检测
    print("\n测试单格子 BDD:")
    empty_0 = gomoku.cell_is_empty(0)
    print(f"  格子0为空 BDD 节点数: {len(empty_0)}")
    
    player_0 = gomoku.cell_is_player(0)
    print(f"  格子0是我方 BDD 节点数: {len(player_0)}")
    
    print("\n测试落子转移 (不进行完整验证):")
    # 只测试单次落子
    trans = gomoku.build_move_transition(gomoku.bdd.true, 40, is_player=True)
    print(f"  中心落子转移 BDD 节点数: {len(trans)}")
    
    print("\n✅ 9×9 五子棋基础测试通过")
    print("完整验证需要在 HPC 上运行")

