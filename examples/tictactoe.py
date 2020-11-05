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
        self.reset()

    @property
    def to_play(self):
        return self._to_play

    def legal_actions(self):
        return [i for i, cell in enumerate(self.board.flatten()) if cell == 0]

    def winner(self):
        return self._winner

    def step(self, action, fake=False):
        self._check_action(action)
        i, j = self._parse_action(action)
        board = self.board.copy()
        board[i, j] = self._to_play
        reward, done = self._check_board(board, i, j)

        if fake:
            return TicTacToeObservation(board), reward, done, {}

        self.board = board
        if done:
            self._winner = reward
        self._to_play *= -1
        return TicTacToeObservation(self.board), reward, done, {}

    def reset(self):
        self.board = np.zeros(shape=(3, 3))
        self._to_play = WHITE
        self._winner = None
        return self.board.copy()

    def render(self, mode='human'):
        mapping = {-1: 'O', 0: ' ', 1: 'X'}
        tokens = [[mapping[cell] for cell in row] for row in self.board]
        print("\n-+-+-\n".join(['|'.join([token for token in row]) for row in tokens]))

    @staticmethod
    def _parse_action(action):
        row = action // 3
        column = action % 3
        return row, column

    def _check_action(self, action):
        legal_actions = self.legal_actions()
        if action not in legal_actions:
            raise ValueError(f"found an illegal action {action}; legal actions are {legal_actions}")

    @staticmethod
    def _check_board(board, i, j):
        possible_winner = board[i, j]
        winner_row = abs(board[i, :].sum()) == 3
        if winner_row:
            return possible_winner, True
        winner_col = abs(board[:, j].sum()) == 3
        if winner_col:
            return possible_winner, True
        if i == j:
            winner_diag = abs(board[[0, 1, 2], [0, 1, 2]].sum()) == 3
            if winner_diag:
                return possible_winner, True
        elif i + j == 2:
            winner_diag = abs(board[[0, 1, 2], [2, 1, 0]].sum()) == 3
            if winner_diag:
                return possible_winner, True
        is_draw = board.flatten().all()
        return 0, is_draw


class TicTacToeObservation:

    def __init__(self, observation):
        self.board = observation

    def __getitem__(self, item):
        return self.board.__getitem__(item)

    def __eq__(self, other):
        return (self.board == other.board).all()

    def __hash__(self):
        return hash((tuple(self.board.flatten())))

    def __str__(self):
        return str(self.board)

    def __repr__(self):
        return str(self)


if __name__ == '__main__':
    game = TicTacToe()
    s1 = MonteCarloTree(game, 1000)
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
