"""
Connect4 Delta-Uniform Self-Play Environment
Mirroring TicTacToeDeltaSelfPlayEnv for VIPER
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class Connect4DeltaSelfPlayEnv(gym.Env):
    """
    Connect4 Delta-Uniform Self-Play Environment
    
    Features:
    1. Maintains two opponent pools: Baseline (MinMax/Random) and Learned (historical policies)
    2. Samples an opponent uniformly at reset
    3. Supports playing as First (1) or Second (-1) player, flipping view accordingly.
    
    Board State:
    - 1: Self
    - -1: Opponent
    - 0: Empty
    """
    
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, baseline_pool=None, learned_pool=None,
                 play_as_o_prob=0.5, sampling_strategy='uniform'):
        super().__init__()
        
        self.ROW_COUNT = 6
        self.COLUMN_COUNT = 7
        
        self.observation_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(6, 7), 
            dtype=np.float32
        )
        
        self.action_space = spaces.Discrete(7)
        
        self.board = None
        self.done = False
        self.winner = None
        
        self.baseline_pool = baseline_pool if baseline_pool is not None else []
        self.learned_pool = learned_pool
        
        self.current_opponent = None
        self.play_as_o = False
        self.play_as_o_prob = play_as_o_prob
        
        self.sampling_strategy = sampling_strategy
        
        self.step_count = 0
        self.episode_count = 0
        self._debug_print_interval = 10000

    def set_opponent_pools(self, baseline_pool=None, learned_pool=None):
        if baseline_pool is not None:
            self.baseline_pool = baseline_pool
        if learned_pool is not None:
            self.learned_pool = learned_pool

    def _sample_opponent(self):
        all_opponents = list(self.baseline_pool)
        if self.learned_pool is not None and len(self.learned_pool) > 0:
            all_opponents.extend(list(self.learned_pool))
            
        if len(all_opponents) == 0:
            return None
            
        return random.choice(all_opponents)

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            
        self.board = np.zeros((self.ROW_COUNT, self.COLUMN_COUNT), dtype=np.float32)
        self.done = False
        self.winner = None
        self.episode_count += 1
        
        self.play_as_o = (np.random.random() < self.play_as_o_prob)
        self.current_opponent = self._sample_opponent()
        
        # If we play as O (Second), opponent (First) moves first
        if self.play_as_o:
            # Opponent is 1 on board
            opponent_action = self._opponent_move()
            if opponent_action is not None:
                self._drop_piece(self.board, opponent_action, 1)
        
        return self._get_observation(), {}

    def step(self, action):
        self.step_count += 1
        if self.step_count % self._debug_print_interval == 0:
             learned = len(self.learned_pool) if self.learned_pool else 0
             print(f"[C4-DELTA] Step {self.step_count}, Ep {self.episode_count}, Pool: {len(self.baseline_pool)} base + {learned} learned")

        if self.done:
             return self._get_observation(), 0, True, False, {'error': 'game_already_done'}

        if not self._is_valid_action(action):
             self.done = True
             return self._get_observation(), -10, True, False, {'illegal_move': True}

        # My marker: if I am O, I am -1 on board. If X, I am 1.
        my_marker = -1 if self.play_as_o else 1
        
        self._drop_piece(self.board, action, my_marker)
        
        if self._check_winner(self.board, my_marker):
            self.done = True
            self.winner = my_marker
            return self._get_observation(), 1, True, False, {'winner': 'self'}
        
        if len(self._get_legal_actions()) == 0:
            self.done = True
            return self._get_observation(), 0, True, False, {'draw': True}
            
        # Opponent move
        opponent_action = self._opponent_move()
        if opponent_action is not None:
             opponent_marker = 1 if self.play_as_o else -1
             self._drop_piece(self.board, opponent_action, opponent_marker)
             
             if self._check_winner(self.board, opponent_marker):
                 self.done = True
                 self.winner = opponent_marker
                 return self._get_observation(), -1, True, False, {'winner': 'opponent'}
             
             if len(self._get_legal_actions()) == 0:
                 self.done = True
                 return self._get_observation(), 0, True, False, {'draw': True}

        return self._get_observation(), 0, False, False, {}

    def _get_observation(self):
        """
        Always return 1 as Self, -1 as Opponent.
        """
        if self.play_as_o:
            return -self.board.copy()
        else:
            return self.board.copy()

    def _opponent_move(self):
        if self.current_opponent is not None:
            # Prepare view for opponent.
            # Opponent always wants to see itself as 1.
            # If we are O (-1), Opp is X (1). Board is correct for Opp.
            # If we are X (1), Opp is O (-1). Board needs flip for Opp.
            
            if self.play_as_o:
                opp_view = self.board.copy()
            else:
                opp_view = -self.board.copy()
            
            try:
                # Some policies might support action_mask. 
                # For Connect4, mask is simply top row is empty.
                # Construct mask: 1 if valid, 0 if invalid (full)
                # mask shape (7,)
                mask = np.array([1 if self.board[self.ROW_COUNT-1][col] == 0 else 0 for col in range(self.COLUMN_COUNT)], dtype=np.int8)
                
                try:
                    action, _ = self.current_opponent.predict(opp_view, deterministic=False, action_masks=mask)
                except TypeError:
                    action, _ = self.current_opponent.predict(opp_view, deterministic=False)
                    
                if self._is_valid_action(action):
                    return action
                else:
                    pass # Fallback
            except Exception as e:
                # print(f"Opponent failed: {e}")
                pass
        
        # Random fallback
        legal = self._get_legal_actions()
        if len(legal) > 0:
            return np.random.choice(legal)
        return None

    def _drop_piece(self, board, col, piece):
        for r in range(self.ROW_COUNT):
             if board[r][col] == 0:
                 board[r][col] = piece
                 return

    def _is_valid_action(self, action):
        if action < 0 or action >= self.COLUMN_COUNT:
            return False
        return self.board[self.ROW_COUNT-1][action] == 0

    def _get_legal_actions(self):
        # returns list of column indices
        return [c for c in range(self.COLUMN_COUNT) if self._is_valid_action(c)]

    def _check_winner(self, board, piece):
        # Check horizontal locations for win
        for c in range(self.COLUMN_COUNT-3):
            for r in range(self.ROW_COUNT):
                if board[r][c] == piece and board[r][c+1] == piece and board[r][c+2] == piece and board[r][c+3] == piece:
                    return True

        # Check vertical locations for win
        for c in range(self.COLUMN_COUNT):
            for r in range(self.ROW_COUNT-3):
                if board[r][c] == piece and board[r+1][c] == piece and board[r+2][c] == piece and board[r+3][c] == piece:
                    return True

        # Check positively sloped diaganols
        for c in range(self.COLUMN_COUNT-3):
            for r in range(self.ROW_COUNT-3):
                if board[r][c] == piece and board[r+1][c+1] == piece and board[r+2][c+2] == piece and board[r+3][c+3] == piece:
                    return True

        # Check negatively sloped diaganols
        for c in range(self.COLUMN_COUNT-3):
            for r in range(3, self.ROW_COUNT):
                if board[r][c] == piece and board[r-1][c+1] == piece and board[r-2][c+2] == piece and board[r-3][c+3] == piece:
                    return True
        return False
        
    def render(self, mode='human'):
        output = "\n"
        output += f"  Playing as: {'O (Second)' if self.play_as_o else 'X (First)'}\n"
        for r in range(self.ROW_COUNT-1, -1, -1):
            row_str = "|"
            for c in range(self.COLUMN_COUNT):
                val = self.board[r][c]
                if val == 1: s = "X"
                elif val == -1: s = "O"
                else: s = " "
                row_str += f" {s} "
            row_str += "|"
            output += row_str + "\n"
        output += "-" * (self.COLUMN_COUNT*3 + 2) + "\n"
        output += "  0  1  2  3  4  5  6\n"
        
        if mode == 'human':
            print(output)
        return output

    def close(self):
        pass
