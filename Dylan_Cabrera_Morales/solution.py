from player import Player
from board import HexBoard
import math

class SmartPlayer(Player):
    def play(self, board : HexBoard) -> tuple:
        # Tu lógica aquí
        depth = 3
        best_move = None
        best_value = -math.inf

        for move in self.get_valid_moves(board):
            new_board = board.clone()
            new_board.place_piece(move[0],move[1],self.player_id)

            value = self.minimax(new_board,depth - 1, False ,-math.inf,math.inf)

            if value > best_value:
                best_value = value
                best_move = move
        
        return best_move

    def get_valid_moves(self , board : HexBoard) -> list :
        moves = []
        for i in range(board.size):
            for j in range(board.size):
                if board.board[i][j] == 0:
                    moves.append((i,j))
        return moves
    
    #Heuristica de conexion
    def conection_score(self , board : HexBoard , player_id : int) -> float:
        score = 0

        for i in range(board.size):
            for j in range(board.size):
                if board.board[i][j] == player_id:
                    score += 2
                elif board.board[i][j] == 0:
                    score += 0.5
        return score
    
    def evaluate(self , board : HexBoard):
        my_score = self.conection_score(board , self.player_id)

        opponent = 2 if self.player_id == 1 else 1
        opp_score = self.conection_score(board,opponent)

        return my_score - opp_score
    
    def minimax(self , board : HexBoard , depth : int , maximizing : bool, alpha : float , beta : float):
        opponent = 2 if self.player_id == 1 else 1

        if board.check_connection(self.player_id):
            return 1000
        
        if board.check_connection(opponent):
            return -1000
        
        if depth == 0:
            return self.evaluate(board)
        
        if maximizing:
            max_eval = -math.inf
            for move in self.get_valid_moves(board):
                new_board = board.clone()
                new_board.place_piece(move[0],move[1],self.player_id)

                eval = self.minimax(new_board , depth - 1, False ,alpha,beta)

                max_eval = max(max_eval,eval)
                alpha = max(alpha , eval)

                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = math.inf
            for move in self.get_valid_moves(board):
                new_board = board.clone()
                new_board.place_piece(move[0],move[1],opponent)

                eval = self.minimax(new_board , depth - 1 , True , alpha , beta)

                min_eval = min(min_eval,eval)
                beta = min(beta,eval)

                if beta <= alpha:
                    break
            return min_eval