#!/usr/bin/env python
# coding: utf-8

# In[4]:

# Contains the core game state representation and rules. NO EXTERNAL LIBRARIES HERE.

EMPTY = ' '
PLAYER_X = 'X'
PLAYER_O = 'O'
DRAW = 'D'
NO_WINNER = ' '
ANY_BOARD = (-1, -1) # Special value for active_board if player can play anywhere

class UltimateTicTacToeState:
    """Represents the state of the Ultimate Tic-Tac-Toe game."""

    def __init__(self):
        """Initializes an empty game state."""
        # board[big_row][big_col][small_row][small_col]
        self.board = [[[EMPTY for _ in range(3)] for _ in range(3)] for _ in range(9)] # Flattened outer board for easier cloning? No, stick to 4D
        self.board = [[[[EMPTY for _ in range(3)] for _ in range(3)] for _ in range(3)] for _ in range(3)]

        # small_board_wins[big_row][big_col] = ' ', 'X', 'O', or 'D' (Draw)
        self.small_board_wins = [[NO_WINNER for _ in range(3)] for _ in range(3)]
        self.overall_win = NO_WINNER
        self.next_player = PLAYER_X
        self.active_board = ANY_BOARD # Start: Player can play anywhere
        self.last_move = None # Store as (br, bc, sr, sc)
        self.history = [] # Optional: track moves

    def clone(self):
        """Manually creates a deep copy of the game state."""
        new_state = UltimateTicTacToeState()

        # Manually copy the 4D board list
        new_board = []
        for br in range(3):
            big_row_list = []
            for bc in range(3):
                small_board_list = []
                for sr in range(3):
                     # Use list slicing for shallow copy of the innermost list
                    small_row_list = self.board[br][bc][sr][:]
                    small_board_list.append(small_row_list)
                big_row_list.append(small_board_list)
            new_board.append(big_row_list)
        new_state.board = new_board

        # Manually copy the 2D small_board_wins list
        new_sb_wins = []
        for br in range(3):
            # Use list slicing for shallow copy of the inner list
            new_sb_wins.append(self.small_board_wins[br][:])
        new_state.small_board_wins = new_sb_wins

        new_state.overall_win = self.overall_win
        new_state.next_player = self.next_player
        new_state.active_board = self.active_board # Tuple/primitive, safe to copy directly
        new_state.last_move = self.last_move     # Tuple/primitive, safe to copy directly
        new_state.history = self.history[:]       # Shallow copy of history list
        return new_state

    def get_valid_moves(self):
        """
        Gets a list of valid moves (br, bc, sr, sc) for the current player.
        NO EXTERNAL LIBRARIES.
        """
        moves = []
        if self.overall_win != NO_WINNER:
            return [] # Game over

        if self.active_board == ANY_BOARD:
            # Player can play in any small board that is not won/drawn
            for br in range(3):
                for bc in range(3):
                    if self.small_board_wins[br][bc] == NO_WINNER:
                        for sr in range(3):
                            for sc in range(3):
                                if self.board[br][bc][sr][sc] == EMPTY:
                                    moves.append((br, bc, sr, sc))
        else:
            # Player must play in the active board
            br, bc = self.active_board
            # Check if the mandatory board is actually playable
            if self.small_board_wins[br][bc] == NO_WINNER:
                 for sr in range(3):
                    for sc in range(3):
                        if self.board[br][bc][sr][sc] == EMPTY:
                            moves.append((br, bc, sr, sc))
            else:
                # This case should technically not happen if apply_move logic is correct
                # If the target board IS finished, the previous apply_move should have set active_board to ANY_BOARD
                # If it does happen (e.g., initial state error), fall back to ANY_BOARD logic
                # print("Warning: Active board is finished, allowing play anywhere.") # Optional debug
                return self.get_valid_moves_any_board()


        # Fallback if no moves found in the specific board (e.g., board full but not marked drawn yet)
        if not moves and self.active_board != ANY_BOARD and self.small_board_wins[self.active_board[0]][self.active_board[1]] == NO_WINNER:
             # This might happen if the board became full JUST NOW, but win check hasn't run?
             # Or if the logic sending to a full board was flawed. Allow play anywhere.
             # print("Warning: No moves in active board, allowing play anywhere.") # Optional debug
             return self.get_valid_moves_any_board()

        # Final check: if after all that, moves is empty, but game not won, it's a draw situation
        # This check is implicitly handled if get_valid_moves is empty and overall_win == NO_WINNER

        return moves

    def get_valid_moves_any_board(self):
        """Helper to get moves when play is allowed anywhere."""
        moves = []
        for br_any in range(3):
            for bc_any in range(3):
                if self.small_board_wins[br_any][bc_any] == NO_WINNER:
                    for sr_any in range(3):
                        for sc_any in range(3):
                            if self.board[br_any][bc_any][sr_any][sc_any] == EMPTY:
                                moves.append((br_any, bc_any, sr_any, sc_any))
        return moves

    def _check_small_board_winner(self, br, bc):
        """
        Checks the winner of a single small board (br, bc).
        Returns 'X', 'O', 'D' (Draw), or ' ' (No Winner).
        NO EXTERNAL LIBRARIES.
        """
        board = self.board[br][bc]
        # Check rows, cols, diags for 'X' or 'O'
        for p in [PLAYER_X, PLAYER_O]:
            # Rows
            for r in range(3):
                if board[r][0] == p and board[r][1] == p and board[r][2] == p:
                    return p
            # Cols
            for c in range(3):
                if board[0][c] == p and board[1][c] == p and board[2][c] == p:
                    return p
            # Diagonals
            if board[0][0] == p and board[1][1] == p and board[2][2] == p:
                return p
            if board[0][2] == p and board[1][1] == p and board[2][0] == p:
                return p

        # Check for Draw (board full, no winner)
        is_full = True
        for r in range(3):
            for c in range(3):
                if board[r][c] == EMPTY:
                    is_full = False
                    break
            if not is_full:
                break
        if is_full:
            return DRAW

        return NO_WINNER # No winner yet

    def _check_overall_winner(self):
        """
        Checks the overall winner based on small_board_wins.
        Returns 'X', 'O', 'D' (Draw), or ' ' (No Winner).
        NO EXTERNAL LIBRARIES.
        """
        board = self.small_board_wins
        # Check rows, cols, diags for 'X' or 'O' (ignore 'D' for winning lines)
        for p in [PLAYER_X, PLAYER_O]:
             # Rows
            for r in range(3):
                if board[r][0] == p and board[r][1] == p and board[r][2] == p:
                    return p
            # Cols
            for c in range(3):
                if board[0][c] == p and board[1][c] == p and board[2][c] == p:
                    return p
            # Diagonals
            if board[0][0] == p and board[1][1] == p and board[2][2] == p:
                return p
            if board[0][2] == p and board[1][1] == p and board[2][0] == p:
                return p

        # Check for Draw (all small boards finished, no winner)
        is_full = True
        for r in range(3):
            for c in range(3):
                if board[r][c] == NO_WINNER:
                    is_full = False
                    break
            if not is_full:
                break
        if is_full:
            return DRAW # Overall draw if all small boards are decided and no winner

        return NO_WINNER

    def apply_move(self, move):
        """
        Applies a move (br, bc, sr, sc) to the state.
        Returns a NEW state object with the move applied.
        Checks for validity but relies on get_valid_moves for primary filtering.
        NO EXTERNAL LIBRARIES.
        """
        if not move or len(move) != 4:
             # Handle potential invalid input if called directly
             print("Error: Invalid move format provided to apply_move")
             return self # Or raise an error

        br, bc, sr, sc = move

        # Basic validity checks (should be caught by get_valid_moves but good defense)
        if self.overall_win != NO_WINNER: return self # Game ended
        if self.board[br][bc][sr][sc] != EMPTY: return self # Cell occupied
        if self.small_board_wins[br][bc] != NO_WINNER: return self # Board finished
        if self.active_board != ANY_BOARD and self.active_board != (br, bc):
             # If forced play, check if move is in the right board
             # Need to handle the case where active_board WAS finished => ANY_BOARD
             active_br, active_bc = self.active_board
             if self.small_board_wins[active_br][active_bc] != NO_WINNER:
                  # The required board was finished, player should have been able to play anywhere
                  # So, this move IS valid IF the target board (br, bc) is playable
                  if self.small_board_wins[br][bc] != NO_WINNER: return self # Target board also finished
                  # Allow the move
             else:
                  # Required board was playable, but move is elsewhere
                  return self # Invalid move

        # Create a new state object for the result
        new_state = self.clone()

        # --- Apply the move ---
        new_state.board[br][bc][sr][sc] = new_state.next_player
        new_state.last_move = move
        # new_state.history.append(move) # Optional history tracking

        # --- Update small board status ---
        small_winner = new_state._check_small_board_winner(br, bc)
        new_state.small_board_wins[br][bc] = small_winner

        # --- Update overall game status ---
        # Only check if the small board win status actually changed *or* if it was the last empty small board
        # For simplicity, we can check every time a small board is potentially decided
        if small_winner != NO_WINNER:
            new_state.overall_win = new_state._check_overall_winner()

        # --- Determine the next active board ---
        next_active_br, next_active_bc = sr, sc
        if new_state.overall_win == NO_WINNER: # Only set next board if game not over
            if new_state.small_board_wins[next_active_br][next_active_bc] != NO_WINNER:
                # If the target board is already won/drawn, next player plays anywhere
                new_state.active_board = ANY_BOARD
            else:
                # Otherwise, next player plays in the specific board
                new_state.active_board = (next_active_br, next_active_bc)
        else:
             new_state.active_board = ANY_BOARD # Game over, doesn't matter

        # --- Switch player ---
        new_state.next_player = PLAYER_O if new_state.next_player == PLAYER_X else PLAYER_X

        # --- Final Draw Check ---
        # If no moves are possible for the next player AND the game is not won, it's a draw
        if new_state.overall_win == NO_WINNER and not new_state.get_valid_moves():
            new_state.overall_win = DRAW

        return new_state

    def is_terminal(self):
        """Checks if the game has ended."""
        return self.overall_win != NO_WINNER



# In[5]:


# Define infinity without math module
INFINITY = float('inf') # Or a very large number like 999999 if float('inf') is disliked

class AIPlayer:
    """Base class for different AI implementations."""
    def __init__(self, player, opponent, search_depth=3):
        self.player = player         # 'X' or 'O' that this AI plays as
        self.opponent = opponent     # 'O' or 'X'
        self.search_depth = search_depth
        self.nodes_explored = 0

    def find_best_move(self, state):
        """Must be implemented by subclasses."""
        raise NotImplementedError

    def evaluate(self, state):
        """
        Evaluates the game state from the perspective of self.player.
        Positive score = good for self.player, Negative = good for opponent.
        """
        if state.overall_win == self.player:
            return 1000 # Win
        elif state.overall_win == self.opponent:
            return -1000 # Loss
        elif state.overall_win == DRAW:
            return 0 # Draw
        else:
            # Heuristic: Count small board wins + potential wins (simple version)
            score = 0
            for br in range(3):
                for bc in range(3):
                    if state.small_board_wins[br][bc] == self.player:
                        score += 10
                    elif state.small_board_wins[br][bc] == self.opponent:
                        score -= 10
            # Add potential wins on the main board? (More complex heuristic)
            # Add control of center small board?
            return score

    def _order_moves_mrv(self, state, moves):
        """
        Orders moves using Minimum Remaining Values (MRV) heuristic.
        Prioritizes moves that leave the opponent with fewer options.
        Returns a sorted list of moves.
        """
        if not moves:
            return []

        move_scores = []
        for move in moves:
            try:
                next_state = state.apply_move(move)
                 # Count opponent's moves in the resulting state
                opponent_moves = next_state.get_valid_moves()
                move_scores.append((len(opponent_moves), move))
            except Exception as e:
                 # Handle cases where apply_move might fail unexpectedly
                 # print(f"Warning: Error evaluating MRV for move {move}: {e}")
                 move_scores.append((999, move)) # Penalize problematic moves

        # Sort by opponent's move count (ascending)
        move_scores.sort(key=lambda x: x[0])
        return [move for score, move in move_scores]

# --- Minimax AI ---
class MinimaxAI(AIPlayer):
    """AI using basic Minimax algorithm."""

    def find_best_move(self, state):
        self.nodes_explored = 0
        possible_moves = state.get_valid_moves()
        if not possible_moves: return None

        best_move = possible_moves[0] # Default move
        best_score = -INFINITY if state.next_player == self.player else INFINITY

        is_maximizing = (state.next_player == self.player)

        for move in possible_moves:
            self.nodes_explored += 1
            next_state = state.apply_move(move)
            score = self.minimax(next_state, self.search_depth - 1, not is_maximizing)

            if is_maximizing:
                if score > best_score:
                    best_score = score
                    best_move = move
            else: # Minimizing player (should not happen if AI is called on its turn, but for safety)
                if score < best_score:
                    best_score = score
                    best_move = move

        # print(f"Minimax explored {self.nodes_explored} nodes.")
        return best_move

    def minimax(self, state, depth, is_maximizing):
        self.nodes_explored += 1
        if depth == 0 or state.is_terminal():
            return self.evaluate(state)

        possible_moves = state.get_valid_moves()
        if not possible_moves:
            return self.evaluate(state) # Evaluate terminal/stuck state

        if is_maximizing:
            max_eval = -INFINITY
            for move in possible_moves:
                next_state = state.apply_move(move)
                eval_score = self.minimax(next_state, depth - 1, False)
                max_eval = max(max_eval, eval_score)
            return max_eval
        else: # Minimizing
            min_eval = INFINITY
            for move in possible_moves:
                next_state = state.apply_move(move)
                eval_score = self.minimax(next_state, depth - 1, True)
                min_eval = min(min_eval, eval_score)
            return min_eval


# --- CSP AI (Backtracking + Forward Checking + MRV) ---
class CspAI(AIPlayer):
    """AI using Backtracking Search with Forward Checking and MRV."""

    def find_best_move(self, state):
        self.nodes_explored = 0
        possible_moves = state.get_valid_moves()
        if not possible_moves: return None

        # Apply MRV heuristic for move ordering
        ordered_moves = self._order_moves_mrv(state, possible_moves)

        best_move = ordered_moves[0] # Default move
        best_score = -INFINITY # AI always maximizes its own score perspective

        for move in ordered_moves:
            self.nodes_explored += 1
            next_state = state.apply_move(move)

            # Basic Forward Checking: Is the next state immediately losing?
            # (Opponent has a winning move right after this move)
            if self._is_immediately_losing(next_state):
                continue # Prune this move

            # Basic Forward Checking: Does the opponent have any moves at all?
            if not next_state.get_valid_moves() and not next_state.is_terminal():
                # If opponent has no moves but game not over, it implies a draw was forced
                # This state is valid, but may not be desirable unless score reflects draw
                 pass # Don't prune, let evaluation handle it

            score = self.backtrack_search(next_state, self.search_depth - 1, False) # Opponent's turn next (minimizing)

            # Since we are maximizing from the current state's perspective
            if score > best_score:
                best_score = score
                best_move = move

        # print(f"CSP AI explored {self.nodes_explored} nodes.")
        return best_move

    def backtrack_search(self, state, depth, is_maximizing):
        """Recursive backtracking search (similar to minimax)."""
        self.nodes_explored += 1
        if depth == 0 or state.is_terminal():
            return self.evaluate(state) # Evaluate the leaf/terminal node

        possible_moves = state.get_valid_moves()
        # Forward Checking: If no moves possible from here, evaluate the state
        if not possible_moves:
             return self.evaluate(state)

        # MRV ordering within recursion (optional, adds overhead)
        # ordered_moves = self._order_moves_mrv(state, possible_moves)
        ordered_moves = possible_moves # Keep it simpler here

        if is_maximizing:
            max_eval = -INFINITY
            for move in ordered_moves:
                next_state = state.apply_move(move)
                # Forward Checking (simplified): check if opponent has moves
                if not next_state.get_valid_moves() and not next_state.is_terminal():
                     eval_score = self.evaluate(next_state) # Reached a drawn state
                else:
                     eval_score = self.backtrack_search(next_state, depth - 1, False)
                max_eval = max(max_eval, eval_score)
            return max_eval
        else: # Minimizing
            min_eval = INFINITY
            for move in ordered_moves:
                next_state = state.apply_move(move)
                 # Forward Checking (simplified)
                if not next_state.get_valid_moves() and not next_state.is_terminal():
                     eval_score = self.evaluate(next_state)
                else:
                    eval_score = self.backtrack_search(next_state, depth - 1, True)
                min_eval = min(min_eval, eval_score)
            return min_eval

    def _is_immediately_losing(self, state):
        """Forward Checking helper: Can the opponent win in the very next turn?"""
        if state.is_terminal(): return False # Game already over

        opponent_moves = state.get_valid_moves()
        for opp_move in opponent_moves:
            next_next_state = state.apply_move(opp_move)
            if next_next_state.overall_win == self.opponent:
                return True # Opponent has a winning reply
        return False


# --- Hybrid AI (Alpha-Beta + CSP techniques) ---
class HybridAI(AIPlayer):
    """AI using Alpha-Beta Pruning combined with CSP Forward Checking & MRV."""

    def __init__(self, player, opponent, search_depth=3, use_mrv=True):
         super().__init__(player, opponent, search_depth)
         self.use_mrv = use_mrv

    # --- ADD THIS METHOD ---
    def _is_immediately_losing(self, state):
        """Forward Checking helper: Can the opponent win in the very next turn?
           'state' is the state *after* the AI made its hypothetical move.
           So, state.next_player is the opponent.
        """
        if state.is_terminal(): return False # Game already over

        opponent_moves = state.get_valid_moves() # Moves for the opponent
        for opp_move in opponent_moves:
            # Simulate opponent's move
            next_next_state = state.apply_move(opp_move)
            # Check if the opponent (state.next_player) won immediately
            if next_next_state.overall_win == state.next_player:
                return True # Opponent has a winning reply immediately after AI's move
        return False
    # --- END OF ADDED METHOD ---

    # --- ADD THIS METHOD --- (Already existed, but ensure it's present)
    def _is_immediately_winning(self, state):
        """Forward Checking helper: Can self.player win in the very next turn?
           'state' is the state *after* the opponent made their hypothetical move.
           So, state.next_player is the AI player.
        """
        if state.is_terminal(): return False
        my_moves = state.get_valid_moves() # Moves for the AI player
        for my_move in my_moves:
            next_next_state = state.apply_move(my_move)
            if next_next_state.overall_win == state.next_player: # Check if AI won
                return True
        return False
    # --- END OF ADDED METHOD ---


    def find_best_move(self, state):
        self.nodes_explored = 0
        possible_moves = state.get_valid_moves()
        if not possible_moves: return None

        if self.use_mrv:
            ordered_moves = self._order_moves_mrv(state, possible_moves)
        else:
            ordered_moves = possible_moves

        best_move = ordered_moves[0] # Default move
        best_score = -INFINITY
        alpha = -INFINITY
        beta = INFINITY

        # is_maximizing is True because the AI is making a move from the current state
        for move in ordered_moves:
            self.nodes_explored += 1
            next_state = state.apply_move(move)

            # --- CSP Integration: Forward Checking ---
            # Use the correctly defined method here:
            if self._is_immediately_losing(next_state): # Check if opponent wins right after
                # print(f"FC Pruning move {move} - leads to immediate loss")
                continue # Prune this move as it leads to an immediate loss

            # Check if opponent has any moves (basic consistency)
            if not next_state.get_valid_moves() and not next_state.is_terminal():
                 # Leads to an immediate draw/stalemate for opponent. Evaluate this state directly.
                 score = self.evaluate(next_state)
                 # print(f"FC Found draw state after move {move}")
            else:
                # Recurse using Alpha-Beta
                score = self.alpha_beta_search(next_state, self.search_depth - 1, alpha, beta, False) # Opponent minimizes next

            if score > best_score:
                best_score = score
                best_move = move

            alpha = max(alpha, best_score)
            # Beta cutoff check at the top level (less common but possible)
            if beta <= alpha:
               break # Already found a move as good as or better than the opponent's best alternative

        # print(f"Hybrid AI explored {self.nodes_explored} nodes.")
        return best_move

    def alpha_beta_search(self, state, depth, alpha, beta, is_maximizing):
        self.nodes_explored += 1

        if depth == 0 or state.is_terminal():
            # Constraint Optimization: Adjust eval based on depth (simple version)
            score = self.evaluate(state)
            if state.is_terminal():
                 # Check if player attribute exists before accessing
                 player_attr = getattr(self, 'player', None) # Get self.player safely
                 if player_attr:
                      if state.overall_win == self.player: score += depth # Favor faster wins (higher score closer to root)
                      elif state.overall_win == self.opponent: score -= depth # Favor slower losses (less negative score closer to root)
            return score


        possible_moves = state.get_valid_moves()
        if not possible_moves:
             return self.evaluate(state) # Evaluate terminal/stuck state

        # Optional MRV ordering within recursion (can add overhead)
        # if self.use_mrv:
        #     ordered_moves = self._order_moves_mrv(state, possible_moves)
        # else:
        #     ordered_moves = possible_moves
        ordered_moves = possible_moves # Keep simple


        if is_maximizing: # AI player (self.player) is maximizing
            max_eval = -INFINITY
            for move in ordered_moves:
                next_state = state.apply_move(move)

                # --- CSP Integration: Forward Checking ---
                # Check if opponent wins right after this move
                if self._is_immediately_losing(next_state):
                    # print(f"FC Pruning in max: move leads to loss")
                    # If the move leads to immediate loss, assign worst score?
                    # Or prune? Pruning seems better based on FC concept.
                    continue # Skip this move, it's immediately bad

                # Check if opponent has moves
                if not next_state.get_valid_moves() and not next_state.is_terminal():
                    eval_score = self.evaluate(next_state) # Handle forced draw
                    # print(f"FC Found draw state in max")
                else:
                    eval_score = self.alpha_beta_search(next_state, depth - 1, alpha, beta, False) # Call for minimizing player

                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break # Beta cutoff
            return max_eval
        else: # Minimizing (Opponent's turn)
            min_eval = INFINITY
            for move in ordered_moves:
                next_state = state.apply_move(move)

                # --- CSP Integration: Forward Checking ---
                # Check if *I* (the AI player) win immediately after opponent moves this way
                if self._is_immediately_winning(next_state):
                     # print(f"FC Found winning state in min")
                     # If the AI can win right after this opponent move,
                     # the minimizing opponent would likely avoid this.
                     # This path is very good for the AI, so its score will be high.
                     # Evaluate this state directly as a win?
                     # Let's evaluate it normally but add depth bonus
                     # return self.evaluate(next_state) + depth # Return high score immediately?
                     pass # Let the normal evaluation + depth bonus handle this high score

                # Check if AI player has moves
                if not next_state.get_valid_moves() and not next_state.is_terminal():
                     eval_score = self.evaluate(next_state) # Handle forced draw
                     # print(f"FC Found draw state in min")
                else:
                    eval_score = self.alpha_beta_search(next_state, depth - 1, alpha, beta, True) # Call for maximizing player

                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break # Alpha cutoff
            return min_eval

# --- End of ai_players.py ---


# In[6]:


# ONLY GUI LIBRARIES ALLOWED HERE (tkinter is standard)
import tkinter as tk
from tkinter import font as tkFont
from tkinter import messagebox
from tkinter import ttk # For themed widgets like Combobox


CELL_SIZE = 30
SMALL_PAD = 2
BIG_PAD = 5
BOARD_COLOR = "#CCCCCC" # Light grey
ACTIVE_BOARD_COLOR = "#FFFF99" # Light yellow
INACTIVE_BOARD_COLOR = "#E0E0E0" # Lighter grey
WIN_X_COLOR = "#FFCCCC" # Light red
WIN_O_COLOR = "#CCCCFF" # Light blue
DRAW_COLOR = "#DDDDDD" # Grey


class UltimateTicTacToeGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Ultimate Tic-Tac-Toe")
        # self.master.geometry("600x700") # Adjust as needed

        self.game_state = UltimateTicTacToeState()
        self.buttons = [[[[None for _ in range(3)] for _ in range(3)] for _ in range(3)] for _ in range(3)]
        self.small_board_frames = [[None for _ in range(3)] for _ in range(3)]
        self.ai_player = None
        self.human_player = PLAYER_X # Human starts as X by default

        self.cell_font = tkFont.Font(family="Helvetica", size=14, weight="bold")
        self.status_font = tkFont.Font(family="Helvetica", size=12)

        # --- Control Frame ---
        self.control_frame = tk.Frame(master)
        self.control_frame.pack(pady=10)

        # AI Selection
        tk.Label(self.control_frame, text="Opponent AI:", font=self.status_font).grid(row=0, column=0, padx=5)
        self.ai_type_var = tk.StringVar(value="Hybrid")
        self.ai_combo = ttk.Combobox(self.control_frame, textvariable=self.ai_type_var,
                                     values=["Minimax", "CSP", "Hybrid"], state="readonly", width=10)
        self.ai_combo.grid(row=0, column=1, padx=5)
        self.ai_combo.bind("<<ComboboxSelected>>", self.select_ai)

        # Difficulty (Search Depth)
        tk.Label(self.control_frame, text="Difficulty (Depth):", font=self.status_font).grid(row=0, column=2, padx=5)
        self.depth_var = tk.IntVar(value=3) # Default depth
        self.depth_spinbox = tk.Spinbox(self.control_frame, from_=1, to=6, textvariable=self.depth_var, width=5, command=self.select_ai)
        self.depth_spinbox.grid(row=0, column=3, padx=5)

        # Start Game Button
        self.start_button = tk.Button(self.control_frame, text="New Game (Human=X)", command=self.start_new_game, font=self.status_font)
        self.start_button.grid(row=1, column=0, columnspan=2, pady=5)

        # Status Label
        self.status_label = tk.Label(self.control_frame, text="Game Started. X's turn.", font=self.status_font, width=40)
        self.status_label.grid(row=1, column=2, columnspan=2, pady=5)


        # --- Board Frame ---
        self.board_frame = tk.Frame(master, bg=BOARD_COLOR, bd=3, relief=tk.GROOVE)
        self.board_frame.pack(padx=10, pady=10)

        self._create_board_widgets()
        self.select_ai() # Initialize AI based on default selection
        self.update_gui()


    def _create_board_widgets(self):
        """Creates the 9 small boards and 81 buttons."""
        for br in range(3):
            for bc in range(3):
                # Create frame for the small board
                small_frame = tk.Frame(self.board_frame, bg=INACTIVE_BOARD_COLOR, bd=2, relief=tk.GROOVE)
                small_frame.grid(row=br, column=bc, padx=BIG_PAD, pady=BIG_PAD)
                self.small_board_frames[br][bc] = small_frame

                # Create buttons within the small board frame
                for sr in range(3):
                    for sc in range(3):
                        button = tk.Button(small_frame, text=EMPTY, width=2, height=1,
                                           font=self.cell_font,
                                           command=lambda r1=br, c1=bc, r2=sr, c2=sc: self.handle_click(r1, c1, r2, c2))
                        button.grid(row=sr, column=sc, padx=SMALL_PAD, pady=SMALL_PAD)
                        self.buttons[br][bc][sr][sc] = button

    def start_new_game(self):
        """Resets the game state and GUI for a new game."""
        self.game_state = UltimateTicTacToeState()
        self.human_player = PLAYER_X # Reset human player if needed
        self.select_ai() # Re-select AI with current depth setting
        self.update_gui()
        self.status_label.config(text=f"New Game. {self.game_state.next_player}'s turn.")

    def select_ai(self, event=None):
        """Instantiates the selected AI player based on GUI controls."""
        ai_type = self.ai_type_var.get()
        depth = self.depth_var.get()
        opponent = PLAYER_O if self.human_player == PLAYER_X else PLAYER_X

        if ai_type == "Minimax":
            self.ai_player = MinimaxAI(player=opponent, opponent=self.human_player, search_depth=depth)
        elif ai_type == "CSP":
             # MRV is handled inside the CSP AI find_best_move currently
            self.ai_player = CspAI(player=opponent, opponent=self.human_player, search_depth=depth)
        elif ai_type == "Hybrid":
            self.ai_player = HybridAI(player=opponent, opponent=self.human_player, search_depth=depth, use_mrv=True) # MRV enabled for Hybrid
        else:
            self.ai_player = None # Or a default AI

        # print(f"Selected AI: {ai_type} with depth {depth}")


    def handle_click(self, br, bc, sr, sc):
        """Handles a click on a cell button."""
        move = (br, bc, sr, sc)

        # Check if it's human's turn and the move is valid
        if self.game_state.next_player == self.human_player and not self.game_state.is_terminal():
            valid_moves = self.game_state.get_valid_moves()
            if move in valid_moves:
                self.game_state = self.game_state.apply_move(move)
                self.update_gui()

                # Check for game over after human move
                if self.game_state.is_terminal():
                    self.show_game_over()
                else:
                    # If game not over, trigger AI move
                    self.status_label.config(text=f"AI ({self.ai_player.player}) is thinking...")
                    self.master.update_idletasks() # Update GUI before potentially long AI calculation
                    self.trigger_ai_move()
            else:
                 messagebox.showwarning("Invalid Move", "You cannot play in that cell. Check the active board (yellow) or choose an empty cell in a valid board.")
        elif self.game_state.is_terminal():
            self.show_game_over()
        # else: it's AI's turn, ignore click


    def trigger_ai_move(self):
        """Gets the AI's move and applies it."""
        if not self.ai_player or self.game_state.next_player == self.human_player or self.game_state.is_terminal():
            return

        # Get best move from the selected AI
        # This call might take time and freeze the GUI
        ai_move = self.ai_player.find_best_move(self.game_state)

        if ai_move:
            self.game_state = self.game_state.apply_move(ai_move)
            self.update_gui()
            if self.game_state.is_terminal():
                self.show_game_over()
            else:
                 self.status_label.config(text=f"Your turn ({self.human_player}).")
        else:
            # AI couldn't find a move (should only happen in draw/error states)
            if not self.game_state.is_terminal():
                 self.game_state.overall_win = DRAW # Force draw if AI stuck but game not won
                 self.update_gui()
                 self.show_game_over()


    def update_gui(self):
        """Updates the button texts, colors, and active board highlighting."""
        for br in range(3):
            for bc in range(3):
                is_active = (self.game_state.active_board == ANY_BOARD or self.game_state.active_board == (br, bc))
                board_status = self.game_state.small_board_wins[br][bc]
                frame_bg = INACTIVE_BOARD_COLOR

                # Determine frame background color
                if board_status == PLAYER_X:
                    frame_bg = WIN_X_COLOR
                elif board_status == PLAYER_O:
                    frame_bg = WIN_O_COLOR
                elif board_status == DRAW:
                     frame_bg = DRAW_COLOR
                elif is_active and self.game_state.overall_win == NO_WINNER:
                    frame_bg = ACTIVE_BOARD_COLOR
                else:
                    frame_bg = INACTIVE_BOARD_COLOR # Default inactive

                self.small_board_frames[br][bc].config(bg=frame_bg)

                # Update buttons within the frame
                for sr in range(3):
                    for sc in range(3):
                        button = self.buttons[br][bc][sr][sc]
                        cell_value = self.game_state.board[br][bc][sr][sc]
                        button.config(text=cell_value)

                        # Disable buttons in finished small boards or if game over
                        if board_status != NO_WINNER or self.game_state.overall_win != NO_WINNER:
                            button.config(state=tk.DISABLED)
                        else:
                             button.config(state=tk.NORMAL)

        # Update status label if game not over
        if self.game_state.overall_win == NO_WINNER:
             self.status_label.config(text=f"{self.game_state.next_player}'s turn.")
        else:
            self.show_game_over() # Update status via game over message


    def show_game_over(self):
        """Displays the game over message and disables the board."""
        winner = self.game_state.overall_win
        if winner == DRAW:
            message = "Game Over: It's a Draw!"
        else:
            message = f"Game Over: Player {winner} Wins!"

        self.status_label.config(text=message)
        messagebox.showinfo("Game Over", message)

        # Disable all buttons
        for br in range(3):
            for bc in range(3):
                 # No need to check small board win status anymore
                 for sr in range(3):
                     for sc in range(3):
                         self.buttons[br][bc][sr][sc].config(state=tk.DISABLED)

# --- Main Execution ---

if __name__ == "__main__":
    root = tk.Tk()
    app = UltimateTicTacToeGUI(root)
    root.mainloop()

