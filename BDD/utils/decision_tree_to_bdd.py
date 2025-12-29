from dd.autoref import BDD

class DecisionTreeToBDD:
    def __init__(self, bdd: BDD, game: str = 'tic_tac_toe'):
        self.bdd = bdd
        bdd.configure(reordering=True) # 自动优化变量顺序以减小 BDD 大小
        self.game = game
        if self.game == 'tic_tac_toe':
            self._declare_variables_tic_tac_toe()
        else:
            raise ValueError(f"Unsupported game: {game}")
        

    def _declare_variables_tic_tac_toe(self):
        for i in range(9):
            for j in range(2):
                self.bdd.add_var(f'x_{i}_{j}')

    def parse_rule_condition(self, rule_json):
        """
        解析单条决策树规则 返回 conditon BDD
        还没有返回action
        """
        # 构建conditon
        cond_bdd = self.bdd.true
        for ant in rule_json['antecedents']:
            idx, op, val = ant
            # 将每一条数值约束 AND 起来
            clause = self._constraint_to_bdd(idx, op, val)
            # print(f"\n约束条件 X[{idx}] {op} {val}:")
            # print(f"  BDD节点数: {len(clause)}") 

            cond_bdd = cond_bdd & clause
            # print(f"\n累积条件的BDD节点数: {len(cond_bdd)}")
            # print(f"满足条件的状态数: {self.bdd.count(cond_bdd)}")
        print(f"\n累积条件的BDD节点数: {len(cond_bdd)}")
        print(f"满足条件的状态数: {self.bdd.count(cond_bdd)}")
        return cond_bdd

    def parse_rule_action(self, rule_json):
        """
        解析单条决策树规则 返回 action BDD
        """
        pass

    def parse_rule(self, rule_json):
        """
        解析单条决策树规则 返回 conditon BDD 和 action BDD
        """
        cond_bdd = self.parse_rule_condition(rule_json)
        action_bdd = self.parse_rule_action(rule_json)

        # @TODO
        # 应该是 cond_bdd -> action_bdd 
        return cond_bdd & action_bdd

    def build_bdd_from_rule(self, rule_json):
        """
        构建 BDD
        解析每一条json规则 其中的conditon 和 action 并且拼接到一起
        """
        # @TODO
        # return self.parse_rule(rule_json)
    
    
    def dump_bdd(self, bdd_node, output_base_name='bdd_graph'):
        """
        Exports the BDD node to JSON and DOT format for visualization.
        
        Args:
            bdd_node: The BDD node to export.
            output_base_name: Base name for output files (without extension).
        """
        from BDD.utils.visualization import generate_visualization
        
        json_path, dot_path = generate_visualization(self.bdd, bdd_node, output_base_name)
        
        print(f"Generated visualization files:\n  - {json_path}\n  - {dot_path}")
        print("👉 You can view the DOT file using 'Graphviz Preview' in VS Code.")

    def _get_cell_expr(self, index, var_prefix='x'):
        """获取某个格子的 BDD 变量名"""
        return f'{var_prefix}_{index}_0', f'{var_prefix}_{index}_1'
    
    def _constraint_to_bdd(self, index, op, threshold):
        """
        核心逻辑：将数值不等式转换为布尔逻辑
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

if __name__ == '__main__':
    import sys
    import os
    # Add project root to sys.path to allow running this script directly
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

    bdd = BDD()
    bdd.configure(reordering=True) # 自动优化变量顺序以减小 BDD 大小
    dt_to_bdd = DecisionTreeToBDD(bdd)
    print("=== Internal BDD Manager State ===")
    
    # 1. 获取第一条规则用于测试
    test_rule = user_rule_json['rules'][0]
    
    # 2. 生成该规则的 Condition BDD
    print("Generating BDD for rule condition...")
    rule_bdd_node = dt_to_bdd.parse_rule_condition(test_rule)
    
    # 3. 打印节点信息 (它是一个 Function 对象，内部包含指向 CUDD node 的引用)
    print(f"Generated BDD Node: {rule_bdd_node}")
    print(f"Satisfying assignments count: {bdd.count(rule_bdd_node)}")
    
    # 4. 导出可视化
    # Files will be saved in 'BDD/generated_bdds' or 'generated_bdds' with timestamp
    dt_to_bdd.dump_bdd(rule_bdd_node, 'user_rule')

    expr = dt_to_bdd.bdd.to_expr(rule_bdd_node)
    print(expr)
    