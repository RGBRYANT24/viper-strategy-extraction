import unittest
from dd.autoref import BDD
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from BDD.utils.decision_tree_to_bdd import DecisionTreeToBDD
from BDD.utils.visualization import format_bdd_to_logic

class TestLeafVector(unittest.TestCase):
    def setUp(self):
        self.bdd = BDD()
        self.bdd.configure(reordering=True)
        self.dt_to_bdd = DecisionTreeToBDD(self.bdd)

    def test_process_leaf_vector_basic(self):
        # 3 cells: 0, 1, 2
        probs = [0.3, 0.5, 0.2] + [0.0]*6 
        
        leaf_guards = self.dt_to_bdd.process_leaf_vector(probs)
        
        # Check action 0: Should be picked if 0 is empty
        # Action0 = Empty(0)
        act0 = leaf_guards[0]
        # x_0_0=0, x_0_1=0 means Empty(0)
        # In this implementation, Empty is ~v0 & ~v1
        # Check logic string
        logic0 = format_bdd_to_logic(self.bdd, act0)
        print(f"\nAction 0 Guard: {logic0}")
        # Expected: (¬x_0_0 ∧ ¬x_0_1)
        
        # Check action 1: Should be picked if 1 is empty AND 0 is NOT empty
        act1 = leaf_guards[1]
        logic1 = format_bdd_to_logic(self.bdd, act1)
        print(f"Action 1 Guard: {logic1}")
        # Expected: ¬(¬x_0_0 ∧ ¬x_0_1) ∧ (¬x_1_0 ∧ ¬x_1_1)
        # i.e., "0 is occupied/invalid" AND "1 is empty"
        
        # Verify logical correctness
        # Case 1: 0 is empty. Action 0 should be true.
        assign1 = {'x_0_0': False, 'x_0_1': False}
        self.assertTrue(self.bdd.let(assign1, act0) == self.bdd.true)
        self.assertTrue(self.bdd.let(assign1, act1) == self.bdd.false)
        
        # Case 2: 0 occupied (e.g., Player x_0_0=True, x_0_1=False), 1 empty.
        assign2 = {'x_0_0': True, 'x_0_1': False, 'x_1_0': False, 'x_1_1': False}
        self.assertTrue(self.bdd.let(assign2, act0) == self.bdd.false)
        self.assertTrue(self.bdd.let(assign2, act1) == self.bdd.true)

    def test_pruning(self):
        # If we have only 2 cells probability provided for test 
        probs = [0.9, 0.8] + [0.0]*7
        leaf_guards = self.dt_to_bdd.process_leaf_vector(probs)
        
        self.assertIn(0, leaf_guards)
        self.assertIn(1, leaf_guards)
        # Should contain other indices if they are reachable, but let's check basic logic

if __name__ == '__main__':
    unittest.main()
