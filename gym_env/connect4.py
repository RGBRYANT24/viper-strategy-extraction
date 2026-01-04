"""
Connect4 Environment Implementation - Compatible with VIPER
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gym_env.policies.connect4_policies import Connect4MinMaxPolicy, Connect4RandomPolicy

class Connect4Env(gym.Env):
    """
    Connect4 Single Player Environment (vs AI)
    
    State: 6x7 grid
         1: Player (Self)
        -1: Opponent
         0: Empty
    
    Action: Discrete(7) - Column to drop piece
    
    Reward:
        +1: Win
        -1: Loss
         0: Draw/Continue
        -10: Illegal move
    """
    
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, opponent_type='random', minmax_depth=2, play_as_o_prob=0.5):
        super().__init__()
        
        self.ROW_COUNT = 6
        self.COLUMN_COUNT = 7
        self.WINDOW_LENGTH = 4
        
        # Observation space: 6x7 grid
        self.observation_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(6, 7), 
            dtype=np.float32
        )
        
        self.play_as_o_prob = play_as_o_prob
        self.play_as_o = False
        
        self.action_space = spaces.Discrete(7)
        
        self.board = None
        self.done = False
        self.winner = None
        
        self.opponent_type = opponent_type
        if opponent_type == 'minmax':
            self.opponent_policy = Connect4MinMaxPolicy(
                self.observation_space,
                self.action_space,
                depth=minmax_depth
            )
        else:
            self.opponent_policy = Connect4RandomPolicy(
                self.observation_space,
                self.action_space
            )
            
        self.step_count = 0
        self._debug_print_interval = 5000

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            
        self.board = np.zeros((self.ROW_COUNT, self.COLUMN_COUNT), dtype=np.float32)
        self.done = False
        self.winner = None
        
        self.play_as_o = np.random.rand() < self.play_as_o_prob
        
        if self.play_as_o:
            # Opponent moves first
            # Opponent is 1 on the physiological board (since we are O/-1)
            # But wait, let's keep it simple:
            # Board State:
            #  1: First Player (X)
            # -1: Second Player (O)
            
            # If we are O (player 2), opponent is X (player 1)
            opponent_action = self._opponent_move()
            if opponent_action is not None:
                self._drop_piece(self.board, opponent_action, 1) # Opponent is 1 (X)
        
        return self._get_observation(), {}

    def _get_observation(self):
        """
        Return observation from current player's perspective.
        Always return 1 as Self, -1 as Opponent.
        """
        if self.play_as_o:
            # We are O (-1). We want to see ourselves as 1.
            # So if board has -1 (us), we return 1.
            # If board has 1 (opponent), we return -1.
            return -self.board.copy()
        else:
            # We are X (1). We return as is.
            return self.board.copy()

    def step(self, action):
        self.step_count += 1
        
        if self.done:
             return self._get_observation(), 0, True, False, {'error': 'game_already_done'}

        if not self._is_valid_action(action):
             self.done = True
             return self._get_observation(), -10, True, False, {'illegal_move': True}

        # Determine my piece marker on the real board
        my_marker = -1 if self.play_as_o else 1
        
        # My move
        self._drop_piece(self.board, action, my_marker)
        
        # Check if I won
        if self._check_winner(self.board, my_marker):
            self.done = True
            self.winner = my_marker
            return self._get_observation(), 1, True, False, {'winner': 'self'}
        
        # Check draw
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

    def _opponent_move(self):
        """
        Opponent chooses a move.
        The opponent policy expects input where IT is 1 and ENEMY is -1.
        
        If play_as_o is True:
          We are O (-1). Opponent is X (1).
          Board has 1(opp) and -1(us).
          Opponent wants to see itself as 1. So it sees Board as is.
          
        If play_as_o is False:
          We are X (1). Opponent is O (-1).
          Board has 1(us) and -1(opp).
          Opponent wants to see itself as 1. So we flip signs.
        """
        if self.play_as_o:
            # Opponent is X (1). Board is correct for it.
            opp_view = self.board.copy()
        else:
            # Opponent is O (-1). We flip to look like 1.
            opp_view = -self.board.copy()
            
        try:
             # Predict expects (C, R) or something? No, my policy expects (6,7)
             # but let's check input shape.
             # Policy expects shape (6,7) likely.
             action, _ = self.opponent_policy.predict(opp_view)
             if self._is_valid_action(action):
                 return action
        except Exception as e:
             # Fallback
             print(f"Opponent error: {e}")
             pass
        
        legal = self._get_legal_actions()
        if len(legal) > 0:
            return np.random.choice(legal)
        return None

    def _drop_piece(self, board, col, piece):
        # find lowest empty row
        for r in range(self.ROW_COUNT):
             if board[r][col] == 0:
                 board[r][col] = piece
                 return

    def _is_valid_action(self, action):
        if action < 0 or action >= self.COLUMN_COUNT:
            return False
        return self.board[self.ROW_COUNT-1][action] == 0

    def _get_legal_actions(self):
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
        # Just simple ASCII render
        # Flip board upside down to print correctly (row 0 is bottom in logic usually, but here... wait)
        # In my drop_piece logic:
        # for r in range(ROW_COUNT): if board[r][col] == 0: ...
        # So row 0 is the "bottom" -- the first one filled.
        # But when printing, we usually print top row first (index ROW-1).
        
        output = "\n"
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
