import sys
sys.path.insert(0, '/Documents/Juan Carlos/Estudios/Universidad/5º Carrera/TFG Informatica/ImplementacionTFG')

import time

import numpy as np
from gym.spaces import Discrete, Box

from tfg.games import GameEnv, WHITE, BLACK
from tfg.strategies import Minimax, MonteCarloTree
from tfg.util import play


class TicTacToe(GameEnv):

    def __init__(self):
        self.observation_space = Box(BLACK, WHITE, shape=(3, 3), dtype=np.int32)
        self.action_space = Discrete(9)
        self.board = None
        self._to_play = 0
        self._winner = None
        self._move_count = 0
        self.reset()

    @property
    def to_play(self):
        return self._to_play

    def legal_actions(self):
        return [i for i, cell in enumerate(self.board.flatten()) if cell == 0]

    def winner(self):
        return self._winner

    def step(self, action):
        i, j = self._parse_action(action)
        self._check_action(i, j)
        self.board[i, j] = self._to_play
        self._move_count += 1
        reward, done = self._check_board(i, j)

        if done:
            self._winner = reward
        self._to_play *= -1
        info = {'to_play': self._to_play, 'winner': self._winner}
        return self.board.copy(), reward, done, info

    def reset(self):
        self.board = np.zeros(shape=(3, 3))
        self._to_play = WHITE
        self._winner = None
        self._move_count = 0
        return self.board.copy()

    def render(self, mode='human'):
        mapping = {-1: 'O', 0: ' ', 1: 'X'}
        tokens = [[mapping[cell] for cell in row] for row in self.board]
        print("\n-+-+-\n".join(
            ['|'.join([token for token in row]) for row in tokens]
        )+'\n')

    @staticmethod
    def _parse_action(action):
        row = action // 3
        column = action % 3
        return row, column

    def _check_action(self, i, j):
        if self.board[i, j] != 0:
            raise ValueError(f"found an illegal action {i * 3 + j}; "
                             f"legal actions are {self.legal_actions()}")

    def _check_board(self, i, j):
        if self._move_count < 5:
            return 0, False
        possible_winner = self.board[i, j]
        winner_row = abs(self.board[i, :].sum()) == 3
        if winner_row:
            return possible_winner, True
        winner_col = abs(self.board[:, j].sum()) == 3
        if winner_col:
            return possible_winner, True
        if i == j:
            winner_diag = abs(self.board[[0, 1, 2], [0, 1, 2]].sum()) == 3
            if winner_diag:
                return possible_winner, True
        if i + j == 2:
            winner_diag = abs(self.board[[0, 1, 2], [2, 1, 0]].sum()) == 3
            if winner_diag:
                return possible_winner, True
        if self._move_count == 3 * 3:
            # Draw
            return 0, True
        return 0, False


def encode(board):
    encoded = 0x0
    shift = 0
    bits = {0: 0b00, WHITE: 0b01, BLACK: 0b10}
    for row in board:
        for x in row:
            encoded |= bits[x] << shift
            shift += 2
    return encoded


def decode(board):
    decoded = np.zeros(shape=(3, 3))
    bits = [0, WHITE, BLACK]
    shift = 0
    for i in range(3):
        for j in range(3):
            decoded[i, j] = bits[((0b11 << shift) & board) >> shift]
            shift += 2
    return decoded


if __name__ == '__main__':
    game = TicTacToe()
    s1 = MonteCarloTree(game, max_iter=10_000, max_time=5.)
    s2 = Minimax(game)

    now = time.time()
    p1_wins, draws, p2_wins = play(game, s1, s2, games=50, max_workers=5)
    print("Monte Carlo as WHITE")
    print(f" * Monte Carlo wins: {p1_wins}")
    print(f" * Minimax wins: {p2_wins}")
    print(f" * Draws: {draws}")
    print(f"Finished in {time.time() - now} sec")

    now = time.time()
    p1_wins, draws, p2_wins = play(game, s2, s1, games=50, max_workers=5)
    print("Monte Carlo as BLACK")
    print(f" * Monte Carlo wins: {p2_wins}")
    print(f" * Minimax wins: {p1_wins}")
    print(f" * Draws: {draws}")
    print(f"Finished in {time.time() - now} sec")
