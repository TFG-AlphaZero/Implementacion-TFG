import sys
sys.path.insert(0, '/Documents/Juan Carlos/Estudios/Universidad/5º Carrera/TFG Informatica/ImplementacionTFG')

import numpy as np
from gym.spaces import Discrete, Box

from tfg.games import GameEnv, WHITE, BLACK


class ConnectN(GameEnv):

    def __init__(self, n=4, rows=6, cols=7):
        if rows < n and cols < n:
            raise ValueError("invalid board shape and number to connect")
        self.observation_space = Box(BLACK, WHITE, shape=(rows, cols),
                                     dtype=np.int8)
        self.action_space = Discrete(cols)
        self._n = n
        self.board = None
        self.indices = None
        self._to_play = WHITE
        self._winner = None
        self._move_count = 0
        self.reset()

    @property
    def n(self):
        return self._n

    @property
    def to_play(self):
        return self._to_play

    def step(self, action):
        self._check_action(action)
        i, j = self.indices[action], action
        self.board[i, j] = self._to_play
        self.indices[action] += 1
        self._move_count += 1
        reward, done = self._check_board(i, j)

        if done:
            self._winner = reward
        self._to_play *= -1
        info = {'to_play': self._to_play, 'winner': self._winner}
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
        self._move_count = 0
        return self.board.copy()

    def render(self, mode='human'):
        mapping = {-1: '○', 0: ' ', 1: '●'}
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

    def _check_board(self, i, j):
        n = self._n
        if self._move_count < 2 * n - 1:
            return 0, False

        rows, cols = self.observation_space.shape
        possible_winner = self.board[i, j]

        c = 0
        for t in self.board[i]:
            c = c + 1 if t == possible_winner else 0
            if c == self._n:
                return possible_winner, True

        c = 0
        for t in self.board[:, j]:
            c = c + 1 if t == possible_winner else 0
            if c == self._n:
                return possible_winner, True

        d = np.diag(self.board, k=j - i)
        if len(d) >= self._n:
            c = 0
            for t in d:
                c = c + 1 if t == possible_winner else 0
                if c == self._n:
                    return possible_winner, True

        d = np.diag(np.fliplr(self.board), k=(cols - 1 - j) - i)
        if len(d) >= self._n:
            c = 0
            for t in d:
                c = c + 1 if t == possible_winner else 0
                if c == self._n:
                    return possible_winner, True

        if self._move_count == rows * cols:
            # Draw
            return 0, True

        return 0, False


def n_connected_heuristic(n):
    def analyze_line(line):
        s = 0
        t = 0
        prev = None
        count = 0
        zeros = 0
        # Check if aligned tokens are surrounded by opponents tokens
        surrounded = True
        for x in line:
            if x == 0:
                zeros += 1
                # Add remaining
                s += t
                t = 0
                surrounded = False
                if zeros == 2:
                    # Reset if we find two zeros
                    count = 0
                    zeros = 0
            else:
                zeros = 0
                if x == prev:
                    count += 1
                    if count == n:
                        t += x
                    count -= 1
                else:
                    if not surrounded:
                        s += t
                    t = 0
                    surrounded = prev != 0
                    count = 1

            prev = x
        return s

    def heuristic(observation, to_play=None):
        rows, cols = observation.shape
        s = 0
        for i in range(rows):
            s += analyze_line(observation[i])
        for j in range(cols):
            s += analyze_line(observation[:, j])
        for k in range(-rows, cols + 1):
            s += analyze_line(np.diag(observation, k=k))
            s += analyze_line(np.diag(np.fliplr(observation), k=k))
        return s / np.multiply(*observation.shape)

    return heuristic
