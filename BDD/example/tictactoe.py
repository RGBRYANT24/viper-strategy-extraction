from dd.autoref import BDD
import json

class TicTacToeVerifier:
    def __init__(self):
        self.bdd = BDD()
        # 棋盘大小
        self.N = 9
        
        # ---------------------------------------------------------
        # 1. 变量声明
        # ---------------------------------------------------------
        # 我们需要两组变量：当前状态 (x) 和 下一步状态 (y)
        # 每个格子 i 需要 2 个 bit 来表示状态：(v0, v1)
        # 编码约定:
        #   00: Empty (空)
        #   10: Player (我方, 对应数值 > 0.5)
        #   01: Opponent (敌方, 对应数值 <= -0.5)
        #   11: Invalid (非法状态)
        
        self.x_vars = []
        self.y_vars = []
        
        for i in range(self.N):
            xi_0 = f'x_{i}_0'
            xi_1 = f'x_{i}_1'
            yi_0 = f'y_{i}_0'
            yi_1 = f'y_{i}_1'
            
            self.bdd.declare(xi_0, xi_1, yi_0, yi_1)
            self.x_vars.extend([xi_0, xi_1])
            self.y_vars.extend([yi_0, yi_1])
            
            # 设置这一步的替换映射（用于状态转移）
            # Prime map: x -> y
            self.prime_map = {xi_0: yi_0, xi_1: yi_1} 

    def _get_cell_expr(self, index, var_prefix='x'):
        """获取某个格子的 BDD 变量名"""
        return f'{var_prefix}_{index}_0', f'{var_prefix}_{index}_1'

    def _constraint_to_bdd(self, index, op, threshold):
        """
        2. 核心逻辑：将数值不等式转换为布尔逻辑
        编码约定: Empty(00), Player(10), Opponent(01)
        """
        v0, v1 = self._get_cell_expr(index, 'x')
        
        # Player (我方): 10
        is_player = self.bdd.add_expr(f'{v0} & ~{v1}')
        # Opponent (敌方): 01
        is_opponent = self.bdd.add_expr(f'~{v0} & {v1}')
        # Empty (空): 00
        is_empty = self.bdd.add_expr(f'~{v0} & ~{v1}')
        
        # 解析逻辑
        # 注意：这里的逻辑需根据你的 ML 模型训练时的定义调整
        # 假设：Player=1, Empty=0, Opponent=-1
        
        res = self.bdd.false
        
        # 逻辑判断
        # X[i] > 0.5  => 必须是 Player
        if op == '>' and threshold >= 0.5:
            res = is_player
            
        # X[i] <= 0.5 => NOT Player (可以是 Empty 或 Opponent)
        elif op == '<=' and threshold >= 0.5:
            res = ~is_player
            
        # X[i] <= -0.5 => 必须是 Opponent
        elif op == '<=' and threshold <= -0.5:
            res = is_opponent
            
        # X[i] > -0.5 => NOT Opponent (可以是 Empty 或 Player)
        elif op == '>' and threshold >= -0.5:
            res = ~is_opponent
            
        else:
            print(f"Warning: 未处理的约束条件: X[{index}] {op} {threshold}")
            res = self.bdd.true 

        # 排除非法状态 (11)
        valid_state = self.bdd.add_expr(f'~({v0} & {v1})')
        return res & valid_state

    def parse_rule(self, rule_json):
        """
        解析单条决策树规则，返回 (Condition_BDD, Action_Index)
        """
        # 1. 构建 Antecedents (前提条件)
        cond_bdd = self.bdd.true
        for ant in rule_json['antecedents']:
            idx, op, val = ant
            # 将每一条数值约束 AND 起来
            clause = self._constraint_to_bdd(idx, op, val)
            cond_bdd = cond_bdd & clause
            
        action = rule_json['best_action']
        return cond_bdd, action

    def build_move_transition(self, condition_bdd, action_idx):
        """
        构建由于我方策略导致的单步状态转移关系 T(x, y)
        Trans = Condition(x) AND (y_action == Player) AND (其他 y == 其他 x)
        """
        # 1. 目标位置变成 Player (10)
        y_act_0, y_act_1 = self._get_cell_expr(action_idx, 'y')
        target_change = self.bdd.add_expr(f'{y_act_0} & ~{y_act_1}')
        
        # 2. 其他位置保持不变 (Frame Conditions)
        frame_cond = self.bdd.true
        for i in range(self.N):
            if i == action_idx:
                continue
            x0, x1 = self._get_cell_expr(i, 'x')
            y0, y1 = self._get_cell_expr(i, 'y')
            # y_i == x_i  <==> (x0 <-> y0) & (x1 <-> y1)
            eq = self.bdd.add_expr(f'({x0} <-> {y0}) & ({x1} <-> {y1})')
            frame_cond = frame_cond & eq
            
        # 3. 组合
        # Trans = Condition(x) & Change(y) & Frame(x, y)
        return condition_bdd & target_change & frame_cond

    def get_opponent_transition(self):
        """
        构建对手所有可能的合法回应。
        对手可以在任意 Empty 的格子落子。
        T_opp = OR_over_all_empty_cells ( Cell_i is Empty & Next_Cell_i becomes Opponent & Others Unchanged )
        """
        t_opp = self.bdd.false
        
        for i in range(self.N):
            x0, x1 = self._get_cell_expr(i, 'x')
            y0, y1 = self._get_cell_expr(i, 'y')
            
            # 前提：该位置为空 (00)
            is_empty = self.bdd.add_expr(f'~{x0} & ~{x1}')
            
            # 结果：该位置变成 Opponent (01)
            become_opp = self.bdd.add_expr(f'~{y0} & {y1}')
            
            # 保持：其他位置不变
            frame = self.bdd.true
            for j in range(self.N):
                if i == j: continue
                ux0, ux1 = self._get_cell_expr(j, 'x')
                uy0, uy1 = self._get_cell_expr(j, 'y')
                frame = frame & self.bdd.add_expr(f'({ux0} <-> {uy0}) & ({ux1} <-> {uy1})')
            
            move_i = is_empty & become_opp & frame
            t_opp = t_opp | move_i
            
        return t_opp

    def verify_strategy(self, rules_data):
        print("构建策略 BDD...")
        # 1. 构建我方策略的总转移关系
        # Strategy_Trans = (Rule1_Cond & Move1) | (Rule2_Cond & Move2) ...
        strategy_trans = self.bdd.false
        
        for rule in rules_data:
            cond, action = self.parse_rule(rule)
            rule_trans = self.build_move_transition(cond, action)
            strategy_trans = strategy_trans | rule_trans
            
        print(f"策略 BDD 节点数: {len(strategy_trans)}")
        
        # 2. 构建对手的转移关系 (Opponent Physics)
        print("构建对手逻辑 BDD...")
        opp_trans = self.get_opponent_transition()
        
        # 3. 验证演示：给定一个满足 Rule 的初始状态，推演结果
        # 我们使用 Rule 中的第一个作为例子来生成一个满足该 Rule 的状态
        print("\n--- 验证演示 ---")
        example_cond, _ = self.parse_rule(rules_data[0])
        
        # 挑选一个满足条件的状态 (pick satisfying assignment)
        # 注意：pick 返回的是一个字典，比如 {'x_0_0': True, ...}
        start_state_dict = self.bdd.pick(example_cond)
        if start_state_dict is None:
            print("错误：无法找到满足第一条规则的状态（可能是规则逻辑矛盾）")
            return

        # 将 dict 转换回 BDD 节点表示当前具体状态
        current_state_bdd = self.bdd.cube(start_state_dict)
        print("初始状态 (满足规则条件):")
        self.print_board(start_state_dict)
        
        # 4. 循环推演：直到无法继续或达到最大步数
        print("\n--- 开始全盘推演 ---")
        current_bdd = current_state_bdd
        
        for step in range(1, 10): # 最多9步
            print(f"\n=== Round {step} ===")
            
            # --- 我方回合 ---
            print("[Player Turn]")
            # Next = exists x. (Current(x) & Strategy(x, y))
            # 1. 计算当前状态和我方策略的合取
            # 2. 对 x 变量进行存在量化，得到只包含 y 的 BDD
            next_state_y = self.bdd.quantify(current_bdd & strategy_trans, self.x_vars, forall=False)
            
            # 检查是否有合法的下一步
            if next_state_y == self.bdd.false:
                print("我方无路可走 (或已赢/输/平)，推演结束。")
                break
                
            # 将 y 重命名回 x，作为新的当前状态
            rename_dict = {y: x for x, y in zip(self.x_vars, self.y_vars)}
            my_move_state = self.bdd.let(rename_dict, next_state_y)
            
            print("我方落子后的可能状态:")
            count = self.bdd.count(my_move_state)
            print(f"状态总数: {count}")
            # 打印几个示例
            for i, state in enumerate(self.bdd.pick_iter(my_move_state)):
                if i >= 3: break
                self.print_board(state)

            # --- 敌方回合 ---
            print("[Opponent Turn]")
            # Opponent Next = exists x. (MyMoveState(x) & OppTrans(x, y))
            opp_response_y = self.bdd.quantify(my_move_state & opp_trans, self.x_vars, forall=False)
            
            if opp_response_y == self.bdd.false:
                print("对手无路可走 (或已赢/输/平)，推演结束。")
                break

            opp_response_x = self.bdd.let(rename_dict, opp_response_y)
            
            print("对手回应后的所有可能状态:")
            count = self.bdd.count(opp_response_x)
            print(f"状态总数: {count}")
            
            # 打印几个示例
            for i, state in enumerate(self.bdd.pick_iter(opp_response_x)):
                if i >= 3: break
                print(f"可能局面 {i+1}:")
                self.print_board(state)
            
            # 更新状态，准备下一轮
            current_bdd = opp_response_x

    def print_board(self, state_dict):
        """可视化打印棋盘"""
        board = ['?'] * 9
        for i in range(9):
            # 获取 x 变量的值，默认为 False
            b0 = state_dict.get(f'x_{i}_0', False)
            b1 = state_dict.get(f'x_{i}_1', False)
            
            if not b0 and not b1: ch = '.'
            elif b0 and not b1:   ch = 'X' # Player
            elif not b0 and b1:   ch = 'O' # Opponent
            else:                 ch = 'E' # Error
            board[i] = ch
            
        print(f" {board[0]} | {board[1]} | {board[2]} ")
        print("---+---+---")
        print(f" {board[3]} | {board[4]} | {board[5]} ")
        print("---+---+---")
        print(f" {board[6]} | {board[7]} | {board[8]} ")

# ---------------------------------------------------------
# 运行
# ---------------------------------------------------------

# 你提供的规则数据
user_rule_json = {
  "rules": [
    {
      "antecedents": [
        [1, "<=", 0.5],  # Not Player
        [6, ">", -0.5],  # Not Opponent
        [6, ">", 0.5],   # Player (Combined with above implies Player)
        [0, "<=", -0.5], # Opponent
        [3, ">", -0.5],  # Not Opponent
        [5, ">", -0.5],  # Not Opponent
        [8, "<=", -0.5]  # Opponent
      ],
      "best_action": 4
    }
  ]
}

verifier = TicTacToeVerifier()
verifier.verify_strategy(user_rule_json['rules'])