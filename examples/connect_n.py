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
        legal_actions = self.legal_actions()
        if action not in legal_actions:
            raise ValueError(f"found an illegal action {action}; "
                             f"legal actions are {legal_actions}")

    def _check_board(self, board, i, j):
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
    s1 = HumanStrategy(game)
    s2 = MonteCarloTree(game, max_iter=100, reset_tree=False)
    play(game, s1, s2, render=True, print_results=True)
