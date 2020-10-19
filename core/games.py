import abc

import gym
import numpy as np

WHITE = 1
BLACK = -1


class GameEnv(gym.Env, abc.ABC):
    metadata = {'render.modes': ['human']}
    reward_range = (-1, 1)

    @property
    def to_play(self):
        raise NotImplementedError

    def legal_actions(self):
        raise NotImplementedError

    def fake_step(self, action):
        raise NotImplementedError

    def winner(self):
        raise NotImplementedError


class DummyGame(GameEnv):
    board = None
    left = None
    right = None
    scores = None
    _to_play = None

    def __init__(self, n):
        self.board = self._create_board(n)
        # 0 = left, 1 = right
        self.action_space = gym.spaces.Discrete(2)
        # (board, score)
        self.observation_space = gym.spaces.Tuple([
            gym.spaces.Box(0, n, shape=(n,), dtype=np.int32),
            gym.spaces.Box(0, n * n, shape=(2,), dtype=np.int32)
        ])
        self.reset()

    @property
    def to_play(self):
        return self._to_play

    def is_last_move(self):
        return self.left == self.right - 1

    def done(self):
        return self.left >= self.right

    def winner(self):
        if not self.done():
            return None
        else:
            return 1 if self.scores[0] > self.scores[1] else \
                -1 if self.scores[1] > self.scores[0] else \
                    0

    @staticmethod
    def _create_board(n):
        return np.arange(1, n + 1, dtype=np.int32)

    def step(self, action):
        self.board = self.board.copy()
        self.scores = self.scores.copy()
        if action == 0:
            score = self.board[self.left]
            self.board[self.left] = 0
            self.left += 1
        elif action == 1:
            self.right -= 1
            score = self.board[self.right]
            self.board[self.right] = 0
        else:
            raise ValueError(f"action must be 0 (left) or 1 (right) - found {action}")
        self.scores[self._to_play] += score
        self._to_play = (self._to_play + 1) % 2
        reward = 0 if not self.done() else self.winner()
        return DummyGameObservation(self.board, self.scores), reward, self.done(), {}

    def fake_step(self, action):
        board = self.board.copy()
        scores = self.scores.copy()
        if action == 0:
            score = board[self.left]
            board[self.left] = 0
        elif action == 1:
            score = board[self.right - 1]
            board[self.right - 1] = 0
        else:
            raise ValueError(f"action must be 0 (left) or 1 (right) - found {action}")
        scores[self._to_play] += score
        # If it was the last move then it's done
        return DummyGameObservation(board, scores), score, self.is_last_move(), {}

    def reset(self):
        n = len(self.board)
        self.board = self._create_board(n)
        np.random.shuffle(self.board)
        self.left = 0
        self.right = n
        self.scores = np.zeros(2, dtype=np.int32)
        self._to_play = 0
        return DummyGameObservation(self.board, self.scores)

    def render(self, mode='human'):
        if self.done():
            print("GAME ENDED")
            self._print_score()
            return
        board = " ".join(str(n) for n in self.board if n != 0)
        n = len(board)
        print(board)
        if not self.is_last_move():
            print(f"^{' ' * (n - 2)}^")
            print(f"0{' ' * (n - 2)}1")
        else:
            print("^")
            print("0")
        self._print_score()

    def _print_score(self):
        print(f"SCORE: {' - '.join(map(str, self.scores))}")

    def legal_actions(self):
        return [] if self.done() else [0] if self.is_last_move() else [0, 1]


class DummyGameObservation:

    def __init__(self, board, scores):
        self._data = (board, scores)

    def __getitem__(self, item):
        return self._data.__getitem__(item)

    def __eq__(self, other):
        return (self._data[0] == other._data[0]).all() and (self._data[1] == other._data[1]).all()

    def __hash__(self):
        return hash((tuple(self._data[0]), tuple(self._data[1])))

    def __str__(self):
        return str(self._data)

    def __repr__(self):
        return str(self)
