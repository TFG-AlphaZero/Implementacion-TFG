import sys
sys.path.insert(0, '/Documents/Juan Carlos/Estudios/Universidad/5º Carrera/TFG Informatica/ImplementacionTFG')

import numpy as np
from gym.spaces import Discrete, Box

from tfg.games import GameEnv, WHITE, BLACK
from tfg.strategies import MonteCarloTree, HumanStrategy
from tfg.util import play


class ConnectN(GameEnv):

    def __init__(self, n=4, rows=6, cols=7):
        self.observation_space = Box(BLACK, WHITE, shape=(rows, cols),
                                     dtype=np.int8)
        self.action_space = Discrete(cols)
        self._n = n
        self.board = None
        self.indices = None
        self._to_play = WHITE
        self._winner = None
        self._moves = 0
        self.reset()

    @property
    def n(self):
        return self._n

    @property
    def to_play(self):
        return self._to_play

    def step(self, action, fake=False):
        self._check_action(action)
        board = self.board.copy()
        indices = self.indices.copy()
        i, j = indices[action], action
        board[i, j] = self._to_play
        indices[action] += 1
        reward, done = self._check_board(board, i, j)
        to_play = -self._to_play
        info = {'to_play': to_play}

        if fake:
            return board, reward, done, info

        self.board = board
        self.indices = indices
        if done:
            self._winner = reward
        self._to_play = to_play
        self._moves += 1
        return self.board.copy(), reward, done, info

    def legal_actions(self):
        rows, cols = self.observation_space.shape
        return np.arange(0, cols)[self.indices < rows].tolist()

    def winner(self):
        return self._winner

    def reset(self):
        self.board = np.zeros(shape=self.observation_space.shape, dtype=np.int8)
        self.indices = np.zeros(shape=(self.action_space.n,), dtype=np.int8)
        self._to_play = WHITE
        self._winner = None
        self._moves = 0
        return self.board.copy()

    def render(self, mode='human'):
        mapping = {-1: '\033[96mO\033[0m', 0: ' ', 1: '\033[92mO\033[0m'}
        tokens = [[mapping[cell] for cell in row] for row in self.board]
        print(f"Connect {self._n}")
        print("\n".join(
            ['|' + '|'.join([token for token in row]) + '|'
             for row in reversed(tokens)]
        ))
        cols = self.action_space.n
        print('-'.join(['+'] * (cols + 1)))
        print(' ' + ' '.join(['^'] * cols))
        print(' ' + ' '.join(str(i) for i in range(cols)))

    def _check_action(self, action):
        rows, _ = self.observation_space.shape
        if self.indices[action] == rows:
            raise ValueError(f"found an illegal action {action}; "
                             f"legal actions are {self.legal_actions()}")

    # TODO keep one
    # def _check_board(self, board, i, j):
    #     n = self._n
    #     if self._moves < 2 * n - 1:
    #         return 0, False
    #
    #     rows, cols = self.observation_space.shape
    #     possible_winner = board[i, j]
    #
    #     c = 1
    #     # Horizontal right
    #     for k in range(j + 1, min(j + n, cols)):
    #         if board[i, k] != possible_winner:
    #             break
    #         c += 1
    #         if c == n:
    #             return possible_winner, True
    #     # Horizontal left
    #     for k in range(j - 1, max(-1, j - n), -1):
    #         if board[i, k] != possible_winner:
    #             break
    #         c += 1
    #         if c == n:
    #             return possible_winner, True
    #
    #     c = 1
    #     # Vertical up
    #     for k in range(i + 1, min(i + n, rows)):
    #         if board[k, j] != possible_winner:
    #             break
    #         c += 1
    #         if c == n:
    #             return possible_winner, True
    #     # Vertical down
    #     for k in range(i - 1, max(-1, i - n), -1):
    #         if board[k, j] != possible_winner:
    #             break
    #         c += 1
    #         if c == n:
    #             return possible_winner, True
    #
    #     c = 1
    #     # Diagonal right-up
    #     for k in range(1, n):
    #         if i + k >= rows or j + k >= cols:
    #             break
    #         if board[i + k, j + k] != possible_winner:
    #             break
    #         c += 1
    #         if c == n:
    #             return possible_winner, True
    #     # Diagonal left-down
    #     for k in range(1, n):
    #         if i - k < 0 or j - k < 0:
    #             break
    #         if board[i - k, j - k] != possible_winner:
    #             break
    #         c += 1
    #         if c == n:
    #             return possible_winner, True
    #
    #     c = 1
    #     # Diagonal right-down
    #     for k in range(1, n):
    #         if i - k < 0 or j + k >= cols:
    #             break
    #         if board[i - k, j + k] != possible_winner:
    #             break
    #         c += 1
    #         if c == n:
    #             return possible_winner, True
    #     # Diagonal left-up
    #     for k in range(1, n):
    #         if i + k >= rows or j - k < 0:
    #             break
    #         if board[i + k, j - k] != possible_winner:
    #             break
    #         c += 1
    #         if c == n:
    #             return possible_winner, True
    #
    #     # Move has not been counted yet
    #     if self._moves == rows * cols - 1:
    #         # Draw
    #         return 0, True
    #
    #     return 0, False

    def _check_board(self, board, i, j):
        n = self._n
        if self._moves < 2 * n - 1:
            return 0, False

        _, cols = self.observation_space.shape
        possible_winner = board[i, j]

        c = 0
        for t in board[i]:
            c = c + 1 if t == possible_winner else 0
            if c == self._n:
                return possible_winner, True

        c = 0
        for t in board[:, j]:
            c = c + 1 if t == possible_winner else 0
            if c == self._n:
                return possible_winner, True

        d = np.diag(board, k=j - i)
        if len(d) >= self._n:
            c = 0
            for t in d:
                c = c + 1 if t == possible_winner else 0
                if c == self._n:
                    return possible_winner, True

        d = np.diag(np.fliplr(board), k=(cols - 1 - j) - i)
        if len(d) >= self._n:
            c = 0
            for t in d:
                c = c + 1 if t == possible_winner else 0
                if c == self._n:
                    return possible_winner, True

        is_draw = board.flatten().all()
        return 0, is_draw


if __name__ == '__main__':
    game = ConnectN()
    s1 = MonteCarloTree(game, max_iter=800, reset_tree=False)
    s2 = MonteCarloTree(game, max_iter=800, reset_tree=False)
    play(game, s1, s2)
