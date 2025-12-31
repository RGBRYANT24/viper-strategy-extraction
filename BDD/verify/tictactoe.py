import os
import sys
import joblib
import numpy as np
from dd.autoref import BDD
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from BDD.utils.decision_tree_to_bdd import DecisionTreeToBDD

class TicTacToeVerifier:
    def __init__(self, model_path):
        self.bdd = BDD()
        self.bdd.configure(reordering=True)
        self.model_path = model_path
        
        # Define BDD variables
        # x_i_j: Cell i, bit j. 
        # 00: Empty, 10: Player (X), 01: Opponent (O)
        self.var_names = []
        for i in range(9):
            self.bdd.add_var(f'x_{i}_0')
            self.bdd.add_var(f'x_{i}_1')
            self.var_names.extend([f'x_{i}_0', f'x_{i}_1'])
            
        # Turn variable: 1=Player, 0=Opponent
        self.bdd.add_var('turn')
        self.var_names.append('turn')
        
        # Game Over variable
        self.bdd.add_var('game_over')
        self.var_names.append('game_over')

        # Prime variables for transition relation (next state)
        self.prime_map = {v: v + "'" for v in self.var_names}
        for v in self.var_names:
            self.bdd.add_var(v + "'")

    def load_policies(self):
        print(f"Loading model from: {self.model_path}")
        model = joblib.load(self.model_path)
        dt_model = model.best_estimator_ if hasattr(model, 'best_estimator_') else model
        
        converter = DecisionTreeToBDD(self.bdd, game='tic_tac_toe')
        # Note: converter adds its own variables, but we defined ours. 
        # Ideally DecisionTreeToBDD should use existing BDD manager without clearing assuming same vars.
        # It calls _declare_variables_tic_tac_toe which adds vars if not exist. Matches our naming x_i_j.
        
        self.policies = converter.recursive_build(dt_model.tree_, 0, 9)
        print("Policies loaded and converted to BDD.")

    def _create_cell_expr(self, idx, status, prime=False):
        """
        status: 'empty' (00), 'player' (10), 'opponent' (01)
        """
        suffix = "'" if prime else ""
        v0 = f'x_{idx}_0{suffix}'
        v1 = f'x_{idx}_1{suffix}'
        
        if status == 'empty':
            return self.bdd.add_expr(f'~{v0} & ~{v1}')
        elif status == 'player':
            return self.bdd.add_expr(f'{v0} & ~{v1}')
        elif status == 'opponent':
            return self.bdd.add_expr(f'~{v0} & {v1}')
        else:
            raise ValueError(f"Unknown status: {status}")

    def get_initial_state(self):
        # All cells empty, Turn = Player (1), Game Only = False (0)
        expr = 'turn & ~game_over'
        for i in range(9):
            expr += f' & ~x_{i}_0 & ~x_{i}_1'
        return self.bdd.add_expr(expr)

    def _transition_player(self):
        """
        Construct T_player:
        If turn=1 & !game_over:
          For each action i:
            If Policy_i is true:
               Cell_i becomes Player (10)
               Turn becomes 0
               Other cells unchanged
        """
        # Pre-condition
        pre = self.bdd.add_expr('turn & ~game_over')
        
        t_player = self.bdd.false
        
        for action_idx, policy_node in self.policies.items():
            # Action i logic:
            # Condition: Policy_i is satisfied
            # Effect: Cell i := Player, Turn' := 0
            
            # 1. Cell i changes to Player
            effect_cell = self._create_cell_expr(action_idx, 'player', prime=True)
            
            # 2. Others unchanged
            unchanged = self.bdd.true
            for j in range(9):
                if j != action_idx:
                    # x_j_0' <-> x_j_0
                    unchanged &= self.bdd.add_expr(f"(x_{j}_0 <-> x_{j}_0') & (x_{j}_1 <-> x_{j}_1')")
            
            # 3. Turn changes
            effect_turn = self.bdd.add_expr("~turn'")
            
            # 4. Game Over unchanged (handled by check step usually, or here just explicitly false)
            # We separate logic: Move -> Check Game Over.
            # Here we just say game_over' matches game_over (which is 0)
            effect_go = self.bdd.add_expr("~game_over'") # Stay playing for now
            
            action_trans = policy_node & effect_cell & unchanged & effect_turn & effect_go
            t_player |= action_trans
            
        return pre & t_player

    def _transition_opponent(self):
        """
        Construct T_opponent:
        If turn=0 & !game_over:
           Exists i such that Cell_i is Empty:
              Cell_i becomes Opponent (01)
              Turn becomes 1
              Others unchanged
        """
        pre = self.bdd.add_expr('~turn & ~game_over')
        
        t_opp = self.bdd.false
        
        for i in range(9):
            # Try to move in cell i
            # Condition: Cell i is Empty
            is_empty = self._create_cell_expr(i, 'empty', prime=False)
            
            # Effect: Cell i := Opponent
            effect_cell = self._create_cell_expr(i, 'opponent', prime=True)
            
            # Others unchanged
            unchanged = self.bdd.true
            for j in range(9):
                if j != i:
                    unchanged &= self.bdd.add_expr(f"(x_{j}_0 <-> x_{j}_0') & (x_{j}_1 <-> x_{j}_1')")
            
            # Turn changes
            effect_turn = self.bdd.add_expr("turn'")
            
            # Go unchanged
            effect_go = self.bdd.add_expr("~game_over'")
            
            move_i = is_empty & effect_cell & unchanged & effect_turn & effect_go
            t_opp |= move_i
            
        return pre & t_opp

    def _check_game_over_trans(self):
        """
        Transition that sets game_over=1 if win/draw condition met.
        Actually, standard reachability usually merges Move & Check.
        
        Simplified approach:
        The transitions above produce a NEXT state with game_over=0.
        BUT, if that next state is a winning state, it should be marked/handled.
        
        Alternative:
        We define 'Winning States'. If we reach a winning state, next turn we stay in it or set GO=1.
        
        Let's strictly define Safe Property P: "Opponent Checks Win" is false.
        So we just need to reach all states.
        
        We don't strictly need to model 'Game Over' variable if we just check if any reachable state satisfies 'Opponent Win'.
        But to be correct about 'Reachable', valid gameplay stops after a win.
        
        Refined T:
        T_move = T_player | T_opponent
        
        If Current is Win/Full -> Next is Same (Self-loop) OR Game_Over=1.
        Let's assume game stops.
        
        Let's ignore 'game_over' variable for complexity reduction and just find all reachable board configurations allowed by rules.
        A state is "Terminal" if someone won or full. No moves out of terminal states (or self-loop).
        """
        # For simplicity, let's stick to generating moves. 
        # If board is already won, T_player/T_opp should be false (no moves enabled).
        
        # Win Conditions (Current State)
        win_player = self._build_win_condition('player')
        win_opp = self._build_win_condition('opponent')
        board_full = self._build_board_full()
        
        is_terminal = win_player | win_opp | board_full
        
        # If terminal, no moves allowed (or self loop)
        # Let's add self-loop for terminal states so verification fixed-point works
        terminal_self_loop = is_terminal & \
                             self.bdd.add_expr("(turn <-> turn')") & \
                             self.bdd.add_expr("(game_over <-> game_over')") 

        for j in range(9):
            terminal_self_loop &= self.bdd.add_expr(f"(x_{j}_0 <-> x_{j}_0') & (x_{j}_1 <-> x_{j}_1')")
            
        return terminal_self_loop, is_terminal

    def _build_win_condition(self, who):
        # who: 'player' or 'opponent'
        lines = [
            (0,1,2), (3,4,5), (6,7,8), # Rows
            (0,3,6), (1,4,7), (2,5,8), # Cols
            (0,4,8), (2,4,6)           # Diagonals
        ]
        
        win_expr = self.bdd.false
        for l in lines:
            line_expr = self.bdd.true
            for cell_idx in l:
                line_expr &= self._create_cell_expr(cell_idx, who, prime=False)
            win_expr |= line_expr
        return win_expr

    def _build_board_full(self):
        # All cells are NOT empty
        full = self.bdd.true
        for i in range(9):
            is_empty = self._create_cell_expr(i, 'empty', prime=False)
            full &= ~is_empty
        return full

    def verify(self):
        self.load_policies()
        
        print("Building Transition Relations...")
        t_player = self._transition_player()
        t_opp = self._transition_opponent()
        
        term_loop, is_terminal = self._check_game_over_trans()
        
        # Enforce: Moves only allowed if NOT terminal
        t_player = t_player & ~is_terminal
        t_opp = t_opp & ~is_terminal
        
        # Total Transition
        T = t_player | t_opp | term_loop
        
        print("Starting Reachability Analysis...")
        reachable = self.get_initial_state()
        
        step = 0
        while True:
            # Image computation: Next = exists vars. T & Reachable
            # Rename prime -> normal
            
            # Efficient image computation supported by `dd`? 
            # bdd.image(trans, source, qvars, rename)
            # qvars (quantified) are current state vars
            # rename maps prime to curr
            
            prev_reachable = reachable
            
            # Symbolic Step
            # next_states(x') = Exists x. (Reachable(x) AND T(x, x'))
            
            # Note: dd's relational product or image
            # Let's use direct quantification for clarity if performance allows, 
            # or `bdd.quantify`
            
            # 1. Conjunction
            curr_and_trans = reachable & T
            
            # 2. Existential Quantification
            next_state_prime = self.bdd.quantify(curr_and_trans, self.var_names, forall=False)
            
            # 3. Rename prime -> current
            rename_dict = {v + "'": v for v in self.var_names}
            next_states = self.bdd.let(rename_dict, next_state_prime)
            
            # Union
            reachable = reachable | next_states
            
            count = self.bdd.count(reachable)
            # count gives number of satisfying assignments over ALL variables (including ones not in support?)
            # dd count usually counts based on support or declared vars. 
            # Let's trust it usually behaves or check `nvars` arg.
            
            step += 1
            print(f"Step {step}: Reachable States = {count:.0f}")
            
            if next_states <= prev_reachable: # Fixpoint reached
                break
                
        print(f"Fixpoint reached at step {step}.")
        
        # Verification
        print("Checking Safety Property: Opponent Never Wins...")
        win_opp = self._build_win_condition('opponent')
        
        unsafe_states = reachable & win_opp
        
        if unsafe_states == self.bdd.false:
            print("\n✅ VERIFICATION SUCCESS: The strategy is SAFE. Opponent can NEVER win.")
        else:
            print("\n❌ VERIFICATION FAILED: Unsafe states found!")
            print(f"Number of Losing States Reachable: {self.bdd.count(unsafe_states)}")
            
            # Optional: Print one counterexample
            print("Counterexample (Losing State):")
            model = self.bdd.pick(unsafe_states)
            print(model)

if __name__ == "__main__":
    # Path to the model file
    model_file = os.path.join(os.path.dirname(__file__), "../tests/data/viper_mask_ppo_tree_20251118_000500_X-only_iter15_samples25000_depth15_leaves125.joblib")
    
    verifier = TicTacToeVerifier(model_file)
    verifier.verify()