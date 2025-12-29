
import unittest
import sys
import os

# 将项目根目录添加到 python path，以便导入模块
# 假设当前文件在 BDD/tests/ 下，我们需要向上两级找到项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from dd.autoref import BDD
# 注意这里的导入路径，取决于你的项目结构，可能是 BDD.utils... 或直接 utils...
# 既然目录是 VIPER/viper-verifiable-rl-impl/BDD/utils
from BDD.utils.decision_tree_to_bdd import DecisionTreeToBDD

class TestConstraintToBDD(unittest.TestCase):
    def setUp(self):
        self.bdd = BDD()
        self.bdd.configure(reordering=True)
        self.dt_to_bdd = DecisionTreeToBDD(self.bdd)
        # 为方便测试，我们只关注 index=0 的情况
        self.idx = 0
        self.v0 = f'x_{self.idx}_0'
        self.v1 = f'x_{self.idx}_1'

    def test_user_rule_parsing(self):
        """测试用户提供的具体规则解析"""
        rule_json = {
            "antecedents": [
                [1, "<=", 0.5],  # Not Player (Empty/Opp)
                [6, ">", -0.5],  # Not Opponent (Empty/Player)
                [6, ">", 0.5],   # Player
                [0, "<=", -0.5], # Opponent
                [3, ">", -0.5],  # Not Opponent
                [5, ">", -0.5],  # Not Opponent
                [8, "<=", -0.5]  # Opponent
            ],
            "best_action": 4
        }
        
        # 注意：方法名是 parse_rule_conditon
        cond_bdd = self.dt_to_bdd.parse_rule_conditon(rule_json)
        
        # 导出可视化文件
        self.dt_to_bdd.dump_bdd(cond_bdd, 'test_user_rule.dot')
        
        # 验证1: 构造一个完全满足条件的状态
        # 0: Opp(01), 1: Empty(00), 3: Empty(00), 5: Empty(00), 6: Player(10), 8: Opp(01)
        # 未提及的变量: 2, 4, 7 -> 设为 Empty(00)
        valid_assignment = {
            'x_0_0': False, 'x_0_1': True,  # 0: Opp
            'x_1_0': False, 'x_1_1': False, # 1: Empty (<= 0.5)
            'x_2_0': False, 'x_2_1': False, # 2: Any (Empty)
            'x_3_0': False, 'x_3_1': False, # 3: Empty (> -0.5)
            'x_4_0': False, 'x_4_1': False, # 4: Any (Empty)
            'x_5_0': False, 'x_5_1': False, # 5: Empty (> -0.5)
            'x_6_0': True,  'x_6_1': False, # 6: Player (> 0.5)
            'x_7_0': False, 'x_7_1': False, # 7: Any (Empty)
            'x_8_0': False, 'x_8_1': True   # 8: Opp (<= -0.5)
        }
        
        # 检查是否为 True
        # let 返回的是 BDD，如果是 True 节点则 == bdd.true
        res = self.bdd.let(valid_assignment, cond_bdd)
        self.assertEqual(res, self.bdd.true, "构造的合法状态应当通过验证")
        
        # 验证2: 构造一个违反条件的状态
        # 修改 6 为 Empty (违反 > 0.5)
        invalid_assignment = valid_assignment.copy()
        invalid_assignment['x_6_0'] = False
        invalid_assignment['x_6_1'] = False # Empty
        
        res_fail = self.bdd.let(invalid_assignment, cond_bdd)
        self.assertEqual(res_fail, self.bdd.false, "违反由x_6定义的Player条件的状态应当失败")



if __name__ == '__main__':
    unittest.main()
