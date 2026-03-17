from player import Player
from board import HexBoard
import math
import heapq
import time

class SmartPlayer(Player):
    def play(self, board : HexBoard) -> tuple:
        # Tu lógica aquí
        self.start_time = time.time()
        self.time_limit = 4.5
        depth = 1
        best_move = None

        while time.time() - self.start_time < self.time_limit:
            move = self.search(board, depth)

            if move is not None:
                best_move = move
            
            depth += 1
        
        return best_move

    
    def search(self, board : HexBoard , depth):
        best_value = -math.inf
        best_move = None

        moves = self.ordered_moves(board)

        for move in moves:
            if time.time() - self.start_time > self.time_limit:
                break

            new_board = board.clone()
            new_board.place_piece(move[0], move[1], self.player_id)

            value = self.minimax(new_board, depth - 1, False, -math.inf, math.inf)

            if value > best_value:
                best_value = value
                best_move = move
        
        return best_move
    
    def minimax(self , board : HexBoard , depth : int , maximizing : bool, alpha : float , beta : float):
        opponent = 2 if self.player_id == 1 else 1

        if board.check_connection(self.player_id):
            return 10000
        
        if board.check_connection(opponent):
            return -10000
        
        if depth == 0:
            return self.evaluate_dijkstra(board)
        
        if maximizing:
            max_eval = -math.inf
            for move in self.ordered_moves(board):
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
            for move in self.ordered_moves(board):
                new_board = board.clone()
                new_board.place_piece(move[0],move[1],opponent)

                eval = self.minimax(new_board , depth - 1 , True , alpha , beta)

                min_eval = min(min_eval,eval)
                beta = min(beta,eval)

                if beta <= alpha:
                    break
            return min_eval

    def get_valid_moves(self , board : HexBoard) -> list :
        moves = []
        for i in range(board.size):
            for j in range(board.size):
                if board.board[i][j] == 0:
                    moves.append((i,j))
        return moves
    
    def ordered_moves(self , board : HexBoard) -> list:
        moves = self.get_valid_moves(board)
        moves_scores = []

        for move in moves:
            new_board = board.clone()
            new_board.place_piece(move[0],move[1],self.player_id)
            score = self.evaluate(new_board)
            moves_scores.append((score, move))
        
        moves_scores.sort(reverse=True)

        return [m for _,m in moves_scores]
    
    def get_neighbors(self, r, c, size):
        if r % 2 == 0:
            directions = [(0, -1), (0, 1), (-1, -1), (-1, 0), (1, -1), (1, 0)]
        else:
            directions = [(0, -1), (0, 1), (-1, 0), (-1, 1), (1, 0), (1, 1)]

        neighbors = []
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < size and 0 <= nc < size:
                neighbors.append((nr, nc))
        return neighbors
        

    
    def cell_cost(self, board : HexBoard, r, c , player_id):
        val = board.board[r][c]

        if val == player_id:
            return 0
        elif val == 0:
            return 1
        else:
            return 1000
        
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
    
    def dijkstra_distance(self, board : HexBoard, player_id : int):
        size = board.size
        dist = [[math.inf]*size for _ in range(size)]
        pq = []

        if player_id == 1:
            for r in range(size):
                cost = self.cell_cost(board, r, 0, player_id)
                dist[r][0] = cost
                heapq.heappush(pq, (cost, r, 0))
        else:
            for c in range(size):
                cost = self.cell_cost(board, 0, c, player_id)
                dist[0][c] = cost
                heapq.heappush(pq, (cost, 0, c))
        
        while pq:
            cost, r, c = heapq.heappop(pq)

            if player_id == 1 and c == size - 1:
                return cost
            if player_id == 2 and r == size - 1:
                return cost
            
            for nr , nc in self.get_neighbors(r, c, size):
                new_cost = cost + self.cell_cost(board, nr, nc, player_id)
                if new_cost < dist[nr][nc]:
                    dist[nr][nc] = new_cost
                    heapq.heappush(pq, (new_cost, nr, nc))
        
        return math.inf
    
    def evaluate_dijkstra(self , board : HexBoard):
        my_distance = self.dijkstra_distance(board, self.player_id)

        opponent = 2 if self.player_id == 1 else 1
        opp_distance = self.dijkstra_distance(board, opponent)

        return opp_distance - my_distance
    
    def evaluate(self , board : HexBoard):
        my_score = self.conection_score(board , self.player_id)

        opponent = 2 if self.player_id == 1 else 1
        opp_score = self.conection_score(board,opponent)

        return my_score - opp_score