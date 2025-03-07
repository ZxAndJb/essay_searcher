from typing import List
class Solution:
    def snakesAndLadders(self, board: List[List[int]]) -> int:
        board = list(reversed(board))

        def convertlocation(r, c, roll):
            if r%2 == 0:
                current_idx = r*len(board[0]) + c + roll + 1
            else:
                current_idx = r*len(board[0]) -c + roll +1

        def findlocation(n):
            r = (n - 1) // len(board[0])
            c = (n - 1) % len(board[0])
            if r % 2 == 0:
                c = c
            else:
                c = len(board[0]) - 1 - c

            return r, c

        queue = [(0, 0, 0, 0)]
        visited = [[0] * len(board[0]) for i in range(len(board))]
        while len(queue) != 0:
            print(queue)
            r, c, time, isjump = queue.pop(0)

            if board[r][c] != -1 and not isjump:
                nr, nc = findlocation(board[r][c])
                if (nr == len(board) - 1 and nr % 2 == 0 and nc == len(board[0]) - 1) or (
                        nr == len(board) - 1 and nr % 2 == 1 and nc == 0):
                    return time
                if visited[nr][nc] == 0:
                    visited[nr][nc] = 1
                    queue.append((nr, nc, time, 1))
            else:
                for roll in range(1, 7):
                    nr, nc = convertlocation(r, c, roll)
                    if nr >= len(board) or (nr == len(board) - 1 and nr % 2 == 0 and nc == len(board[0]) - 1) or (
                            nr == len(board) - 1 and nr % 2 == 1 and nc == 0):
                        return time + 1
                    else:
                        if visited[nr][nc] == 0:
                            visited[nr][nc] = 1
                            queue.append((nr, nc, time + 1, 0))


s = Solution()
board =[[-1,4,-1],[6,2,6],[-1,3,-1]]
s.snakesAndLadders(board)