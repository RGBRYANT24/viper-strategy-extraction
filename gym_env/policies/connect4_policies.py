
import numpy as np
from stable_baselines3.common.policies import BasePolicy
import math
import random

ROW_COUNT = 6
COLUMN_COUNT = 7
WINDOW_LENGTH = 4
EMPTY = 0
AI_PIECE = 1 # In the policy context, we play as AI_PIECE
PLAYER_PIECE = -1 # Opponent

class Connect4MinMaxPolicy(BasePolicy):
    """
    MinMax Policy for Connect4 with Alpha-Beta pruning and Heuristics.
    Adapted from user provided code.
    """
    def __init__(self, observation_space, action_space, depth=4):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space
        )
        self.depth = depth

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        """
        Predict the best move using Minimax.
        Args:
            observation: Board state. shape (6, 7) or flattened.
                         The policy assumes it is playing as '1'.
        """
        # Handle batch input
        if len(observation.shape) == 1:
            # Check if it's flattened 42 or just 6x7 flattened
             if observation.shape[0] == 42:
                 observation = observation.reshape(1, 6, 7)
             else:
                 raise ValueError(f"Unexpected observation shape: {observation.shape}")
             single_obs = True
        elif len(observation.shape) == 2:
             if observation.shape == (6,7):
                 observation = observation.reshape(1, 6, 7)
                 single_obs = True
             else:
                  # Batch of flattened
                  observation = observation.reshape(-1, 6, 7)
                  single_obs = False
        elif len(observation.shape) == 3:
             single_obs = False
        else:
             single_obs = False

        actions = []
        for board in observation:
            # The board comes in with 1 (self) and -1 (opponent) usually, or custom.
            # We need to make sure the logic matches.
            # User logic: PLAYER_PIECE=1, AI_PIECE=2. 
            # Our Gym Env usually uses 1 and -1.
            # Let's map: 1 -> AI_PIECE(user's 2), -1 -> PLAYER_PIECE(user's 1)
            # Actually, let's just reuse the user's logic but adapt the piece values.
            # In this policy, we are the AI. So we are maximizing.
            
            # For simplicity, let's assume the input board uses 1 for 'current player' (us) and -1 for opponent.
            # We will use 1 as AI_PIECE and -1 as PLAYER_PIECE internally for the minimax logic.
            
            col, score = self.minimax(board, self.depth, -math.inf, math.inf, True)
            
            # If minimax returns None (e.g. game over or weird state), fallback to random valid
            if col is None:
                 valid_locs = self.get_valid_locations(board)
                 if valid_locs:
                     col = random.choice(valid_locs)
                 else:
                     col = 0
            
            actions.append(col)

        actions = np.array(actions)
        if single_obs:
            return actions[0], None
        return actions, None
    
    def _predict(self, observation, deterministic=True):
        action, _ = self.predict(observation, deterministic=deterministic)
        return action

    def forward(self, obs, deterministic=True):
         return self._predict(obs, deterministic)

    # --- Game Logic Helpers ---
    
    def drop_piece(self, board, row, col, piece):
        board[row][col] = piece

    def is_valid_location(self, board, col):
        return board[ROW_COUNT-1][col] == 0

    def get_next_open_row(self, board, col):
        for r in range(ROW_COUNT):
            if board[r][col] == 0:
                return r

    def get_valid_locations(self, board):
        valid_locations = []
        for col in range(COLUMN_COUNT):
            if self.is_valid_location(board, col):
                valid_locations.append(col)
        return valid_locations

    def winning_move(self, board, piece):
        # Check horizontal locations for win
        for c in range(COLUMN_COUNT-3):
            for r in range(ROW_COUNT):
                if board[r][c] == piece and board[r][c+1] == piece and board[r][c+2] == piece and board[r][c+3] == piece:
                    return True

        # Check vertical locations for win
        for c in range(COLUMN_COUNT):
            for r in range(ROW_COUNT-3):
                if board[r][c] == piece and board[r+1][c] == piece and board[r+2][c] == piece and board[r+3][c] == piece:
                    return True

        # Check positively sloped diaganols
        for c in range(COLUMN_COUNT-3):
            for r in range(ROW_COUNT-3):
                if board[r][c] == piece and board[r+1][c+1] == piece and board[r+2][c+2] == piece and board[r+3][c+3] == piece:
                    return True

        # Check negatively sloped diaganols
        for c in range(COLUMN_COUNT-3):
            for r in range(3, ROW_COUNT):
                if board[r][c] == piece and board[r-1][c+1] == piece and board[r-2][c+2] == piece and board[r-3][c+3] == piece:
                    return True
        return False

    def evaluate_window(self, window, piece):
        score = 0
        opp_piece = PLAYER_PIECE 
        if piece == PLAYER_PIECE:
            opp_piece = AI_PIECE

        # window is a list or array
        
        # Convert to list if it's numpy array just in case
        window = list(window)

        if window.count(piece) == 4:
            score += 100
        elif window.count(piece) == 3 and window.count(EMPTY) == 1:
            score += 5
        elif window.count(piece) == 2 and window.count(EMPTY) == 2:
            score += 2

        if window.count(opp_piece) == 3 and window.count(EMPTY) == 1:
            score -= 4

        return score

    def score_position(self, board, piece):
        score = 0
        
        # We need to ensure we are using the correct pieces for evaluation.
        # AI_PIECE should be 1, PLAYER_PIECE should be -1
        
        ## Score center column
        center_array = [int(i) for i in list(board[:, COLUMN_COUNT//2])]
        center_count = center_array.count(piece)
        score += center_count * 3

        ## Score Horizontal
        for r in range(ROW_COUNT):
            row_array = [int(i) for i in list(board[r,:])]
            for c in range(COLUMN_COUNT-3):
                window = row_array[c:c+WINDOW_LENGTH]
                score += self.evaluate_window(window, piece)

        ## Score Vertical
        for c in range(COLUMN_COUNT):
            col_array = [int(i) for i in list(board[:,c])]
            for r in range(ROW_COUNT-3):
                window = col_array[r:r+WINDOW_LENGTH]
                score += self.evaluate_window(window, piece)

        ## Score posiive sloped diagonal
        for r in range(ROW_COUNT-3):
            for c in range(COLUMN_COUNT-3):
                window = [board[r+i][c+i] for i in range(WINDOW_LENGTH)]
                score += self.evaluate_window(window, piece)

        for r in range(ROW_COUNT-3):
            for c in range(COLUMN_COUNT-3):
                window = [board[r+3-i][c+i] for i in range(WINDOW_LENGTH)]
                score += self.evaluate_window(window, piece)

        return score

    def is_terminal_node(self, board):
        return self.winning_move(board, AI_PIECE) or self.winning_move(board, PLAYER_PIECE) or len(self.get_valid_locations(board)) == 0

    def minimax(self, board, depth, alpha, beta, maximizingPlayer):
        valid_locations = self.get_valid_locations(board)
        is_terminal = self.is_terminal_node(board)
        if depth == 0 or is_terminal:
            if is_terminal:
                if self.winning_move(board, AI_PIECE):
                    return (None, 100000000000000)
                elif self.winning_move(board, PLAYER_PIECE):
                    return (None, -10000000000000)
                else: # Game is over, no more valid moves
                    return (None, 0)
            else: # Depth is zero
                return (None, self.score_position(board, AI_PIECE))
        
        if maximizingPlayer:
            value = -math.inf
            column = random.choice(valid_locations)
            for col in valid_locations:
                row = self.get_next_open_row(board, col)
                b_copy = board.copy()
                self.drop_piece(b_copy, row, col, AI_PIECE)
                new_score = self.minimax(b_copy, depth-1, alpha, beta, False)[1]
                if new_score > value:
                    value = new_score
                    column = col
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return column, value

        else: # Minimizing player
            value = math.inf
            column = random.choice(valid_locations)
            for col in valid_locations:
                row = self.get_next_open_row(board, col)
                b_copy = board.copy()
                self.drop_piece(b_copy, row, col, PLAYER_PIECE)
                new_score = self.minimax(b_copy, depth-1, alpha, beta, True)[1]
                if new_score < value:
                    value = new_score
                    column = col
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return column, value

class Connect4RandomPolicy(BasePolicy):
    def __init__(self, observation_space, action_space):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space
        )

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        # Handle simple vs batch
        if len(observation.shape) == 1:
             if observation.shape[0] == 42:
                 observation = observation.reshape(1, 6, 7)
             else:
                  # Assume it's a batch of 1 if shape is weird but we'll see.
                  # For now assume flat.
                  pass
             single_obs = True
        elif len(observation.shape) == 2:
             if observation.shape == (6,7):
                 observation = observation.reshape(1, 6, 7)
                 single_obs = True
             else:
                 observation = observation.reshape(-1, 6, 7)
                 single_obs = False
        else:
             single_obs = False

        actions = []
        for board in observation:
            valid_locs = self.get_valid_locations(board)
            if valid_locs:
                actions.append(random.choice(valid_locs))
            else:
                actions.append(0) # fallback
        
        actions = np.array(actions)
        if single_obs:
            return actions[0], None
        return actions, None

    def _predict(self, observation, deterministic=True):
        action, _ = self.predict(observation, deterministic=deterministic)
        return action
        
    def forward(self, obs, deterministic=True):
         return self._predict(obs, deterministic)

    def get_valid_locations(self, board):
        valid_locations = []
        for col in range(COLUMN_COUNT):
             # check top row
            if board[ROW_COUNT-1][col] == 0:
                valid_locations.append(col)
        return valid_locations
