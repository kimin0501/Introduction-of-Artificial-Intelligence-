import random
import copy
import time

class TeekoPlayer:
    """ An object representation for an AI game player for the game Teeko.
    """
    board = [[' ' for j in range(5)] for i in range(5)]
    pieces = ['b', 'r']

    def __init__(self):
        """ Initializes a TeekoPlayer object by randomly selecting red or black as its
        piece color.
        """
        self.my_piece = random.choice(self.pieces)
        self.opp = self.pieces[0] if self.my_piece == self.pieces[1] else self.pieces[1]
        self.depth_limit = 2
    
    # helper method to get drop phase
    def get_drop_phase(self, state):     
        drop_phase = False
        piece_num = 0
        
        for row in range(5):
            for col in range(5):
                if state[row][col] != ' ':
                    piece_num += 1

        # determine drop phase by counting the block number
        if (piece_num < 8):
            drop_phase = True
            
            return drop_phase
    
    
    def succ(self, state, color):
        # get the drop phase state from helper method
        drop_phase = self.get_drop_phase(state)
        # list to save every possible movements with the form in (row, col, new_row, new_col)
        successors = []

        # possible movement directions
        movements = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        
        # check for every possible movement and add them to successors list when it is drop phase
        if drop_phase == True:
            for row in range(5):
               for col in range(5):
                   if state[row][col] == ' ':
                       successors.append([row, col, row, col])
        
        # check for every possible movement and add them to successors list when it is not drop phase
        else:
            for row in range(5):
                for col in range(5):
                    if state[row][col] == color:
                        for dr, dc in movements:
                            new_row = row + dr
                            new_col = col + dc
                            if 0 <= new_row < 5 and 0 <= new_col < 5 and state[new_row][new_col] == ' ':
                                successors.append([row, col, new_row, new_col])
                    
        return successors

    def make_move(self, state):
        """ Selects a (row, col) space for the next move. You may assume that whenever
        this function is called, it is this player's turn to move.

        Args:
            state (list of lists): should be the current state of the game as saved in
                this TeekoPlayer object. Note that this is NOT assumed to be a copy of
                the game state and should NOT be modified within this method (use
                place_piece() instead). Any modifications (e.g. to generate successors)
                should be done on a deep copy of the state.

                In the "drop phase", the state will contain less than 8 elements which
                are not ' ' (a single space character).

        Return:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

        Note that without drop phase behavior, the AI will just keep placing new markers
            and will eventually take over the board. This is not a valid strategy and
            will earn you no points.
        """

        drop_phase =  self.get_drop_phase(state)   # TODO: detect drop phase

        if not drop_phase:
            # TODO: choose a piece to move and remove it from the board
            # (You may move this condition anywhere, just be sure to handle it)
            #
            # Until this part is implemented and the move list is updated
            # accordingly, the AI will not follow the rules after the drop phase!
            
            # initialize the variables to track the best move
            best_value = float("-inf")
            best_move = None
            alpha = float("-inf")
            beta = float("inf")
            
          

            # traverse all possible successors
            successors = self.succ(state, self.my_piece)
            for move in successors:
                new_state = copy.deepcopy(state)
                
                # apply the move when it is not drop phase
                if not drop_phase:
                    new_state[move[0]][move[1]] = ' '
                    new_state[move[2]][move[3]] = self.my_piece
                
                # evalute the move value by using alpha-beta pruning
                move_val = self.min_value(new_state, 1, alpha, beta)
                
                # update the best move
                if move_val > best_value:
                    best_value = move_val
                    best_move = move

            # return if the best move is found
            if best_move:
                return [(best_move[2], best_move[3]), (best_move[0], best_move[1])]


        # select an unoccupied space randomly
        # TODO: implement a minimax algorithm to play better
        move = []
        (row, col) = (random.randint(0,4), random.randint(0,4))
        while not state[row][col] == ' ':
            (row, col) = (random.randint(0,4), random.randint(0,4))

        # ensure the destination (row,col) tuple is at the beginning of the move list
        move.insert(0, (row, col))
        
        
        
        return move
    
    
    
    def max_value(self, state, depth, alpha, beta):
        # check for the game state
        if (self.game_value(state)) == 1 or (self.game_value(state)) == -1:
            return self.game_value(state)
        
        # check for the depth
        if depth == self.depth_limit:
            return self.heuristic_game_value(state)
        
        # initialize the max value
        value = float("-inf")
        
        # traverse through successors
        successors = self.succ(state, self.my_piece)
        for successor in successors:
            
            # create a new state
            temp_state = copy.deepcopy(state) 
            
            # check for the drop phase and apply the move 
            if self.get_drop_phase(state) == True:
                temp_state[successor[0]][successor[1]] = self.my_piece
            elif self.get_drop_phase(state) == False:
                temp_state[successor[0]][successor[1]] = ' '
                temp_state[successor[2]][successor[3]] = self.my_piece

            # recursive call and update the value
            move_value = self.min_value(temp_state, depth + 1, alpha, beta)
            value = max(value, move_value)
            alpha = max(alpha, value)

            # compare alpha and beta
            if alpha >= beta:
                break
            
        return value
    
    def min_value(self, state, depth, alpha, beta):
        # check for the game state
        if (self.game_value(state)) == 1 or (self.game_value(state)) == -1:
            return self.game_value(state)
        
        # check for the depth
        if depth == self.depth_limit:
            return self.heuristic_game_value(state)
        
        # initialize the min value
        value = float("inf")
        
        # traverse through successors
        successors = self.succ(state, self.opp)
        for successor in successors:
            
            # create a new state
            temp_state = copy.deepcopy(state) 
            
            # check for the drop phase and apply the move 
            if self.get_drop_phase(state) == True:
                temp_state[successor[0]][successor[1]] = self.opp
            elif self.get_drop_phase(state) == False:
                temp_state[successor[0]][successor[1]] = ' '
                temp_state[successor[2]][successor[3]] = self.opp
            
            # recursive call and update the value
            move_value = self.max_value(temp_state, depth + 1, alpha, beta)    
            value = min(value, move_value)
            beta = min(beta, value)
            
            # compare alpha and beta
            if alpha >= beta:
                break
            
        return value

    def opponent_move(self, move):
        """ Validates the opponent's next move against the internal board representation.
        You don't need to touch this code.

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.
        """
        # validate input
        if len(move) > 1:
            source_row = move[1][0]
            source_col = move[1][1]
            if source_row != None and self.board[source_row][source_col] != self.opp:
                self.print_board()
                print(move)
                raise Exception("You don't have a piece there!")
            if abs(source_row - move[0][0]) > 1 or abs(source_col - move[0][1]) > 1:
                self.print_board()
                print(move)
                raise Exception('Illegal move: Can only move to an adjacent space')
        if self.board[move[0][0]][move[0][1]] != ' ':
            raise Exception("Illegal move detected")
        # make move
        self.place_piece(move, self.opp)

    def place_piece(self, move, piece):
        """ Modifies the board representation using the specified move and piece

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

                This argument is assumed to have been validated before this method
                is called.
            piece (str): the piece ('b' or 'r') to place on the board
        """
        if len(move) > 1:
            self.board[move[1][0]][move[1][1]] = ' '
        self.board[move[0][0]][move[0][1]] = piece

    def print_board(self):
        """ Formatted printing for the board """
        for row in range(len(self.board)):
            line = str(row)+": "
            for cell in self.board[row]:
                line += cell + " "
            print(line)
        print("   A B C D E")

    def game_value(self, state):
        """ Checks the current board status for a win condition

        Args:
        state (list of lists): either the current state of the game as saved in
            this TeekoPlayer object, or a generated successor state.

        Returns:
            int: 1 if this TeekoPlayer wins, -1 if the opponent wins, 0 if no winner

        TODO: complete checks for diagonal and box wins
        """
        # check horizontal wins
        for row in state:
            for i in range(2):
                if row[i] != ' ' and row[i] == row[i+1] == row[i+2] == row[i+3]:
                    return 1 if row[i]==self.my_piece else -1

        # check vertical wins
        for col in range(5):
            for i in range(2):
                if state[i][col] != ' ' and state[i][col] == state[i+1][col] == state[i+2][col] == state[i+3][col]:
                    return 1 if state[i][col]==self.my_piece else -1

        # TODO: check \ diagonal wins
        for j in range(2):
            for i in range(2):
                if state[i][j] != ' ' and state[i][j] == state[i + 1][j + 1] == state[i + 2][j + 2] == state[i + 3][j + 3]:
                    return 1 if state[i][j] == self.my_piece else -1
        
        # TODO: check / diagonal wins
        for j in range(2):
            for i in range(3, 5):
                if state[i][j] != ' ' and state[i][j] == state[i - 1][j + 1] == state[i - 2][j + 2] == state[i - 3][j + 3]:
                    return 1 if state[i][col] == self.my_piece else -1
        
        # TODO: check box wins
        for i in range(4):
            for j in range(4):
                if state[i][j] != ' ' and state[i][j] == state[i + 1][j] == state[i][j + 1] == state[i + 1][j + 1]:
                    return 1 if state[i][j] == self.my_piece else -1

        return 0  # no winner yet
    
    def heuristic_game_value(self, state):
        # check for the game status
        state_value = self.game_value(state)
        
        if state_value != 0:
            return state_value
        
        # initialize the weight for the piece
        general_weight = 0.1
        center_bonus = 0.2
        special_position_bonus = 0.1
        
        # initialzie the player score
        player_score = 0.0
        ai_score = 0.0
        
        # traverse the game board to compute the score
        for row in range(5):
            for col in range(5):
                # check for the player score
                if state[row][col] == self.my_piece:
                    player_score += general_weight

                    # additional values for center position
                    if row == 2 and col == 2:
                        player_score += center_bonus

                    # additional values for special position
                    if (row == 2 and col == 1) or (row == 1 and col == 2):
                        player_score += special_position_bonus

                # check for the opposite score
                elif state[row][col] == self.opp:
                    ai_score += general_weight
                    
                     # additional values for center position
                    if row == 2 and col == 2:
                        ai_score += center_bonus

                    # additional values for special position
                    if (row == 2 and col == 1) or (row == 1 and col == 2):
                        ai_score += special_position_bonus

        # compute heuristic value
        heuristic_value = player_score - ai_score

        return heuristic_value
        
        

############################################################################
#
# THE FOLLOWING CODE IS FOR SAMPLE GAMEPLAY ONLY
#
############################################################################
def main():
    print('Hello, this is Samaritan')
    ai = TeekoPlayer()
    piece_count = 0
    turn = 0

    # drop phase
    while piece_count < 8 and ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved at "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                player_move = input("Move (e.g. B3): ")
                while player_move[0] not in "ABCDE" or player_move[1] not in "01234":
                    player_move = input("Move (e.g. B3): ")
                try:
                    ai.opponent_move([(int(player_move[1]), ord(player_move[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        piece_count += 1
        turn += 1
        turn %= 2

    # move phase - can't have a winner until all 8 pieces are on the board
    while ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved from "+chr(move[1][1]+ord("A"))+str(move[1][0]))
            print("  to "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                move_from = input("Move from (e.g. B3): ")
                while move_from[0] not in "ABCDE" or move_from[1] not in "01234":
                    move_from = input("Move from (e.g. B3): ")
                move_to = input("Move to (e.g. B3): ")
                while move_to[0] not in "ABCDE" or move_to[1] not in "01234":
                    move_to = input("Move to (e.g. B3): ")
                try:
                    ai.opponent_move([(int(move_to[1]), ord(move_to[0])-ord("A")),
                                    (int(move_from[1]), ord(move_from[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        turn += 1
        turn %= 2

    ai.print_board()
    if ai.game_value(ai.board) == 1:
        print("AI wins! Game over.")
    else:
        print("You win! Game over.")


if __name__ == "__main__":
    main()