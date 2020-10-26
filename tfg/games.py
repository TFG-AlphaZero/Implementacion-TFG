import abc

import gym
import numpy as np

WHITE = 1
BLACK = -1


class GameEnv(gym.Env, abc.ABC):
    """
    Base class for two player games. Extends gym's Env base class to add the capacity to know the result of a certain
    step without actually taking it. It also adds methods to know all the legal actions in the current state,
    a subset of the action space, and to know the winner in the current state.
    """
    metadata = {'render.modes': ['human']}
    reward_range = (-1, 1)

    @property
    def to_play(self):
        """
        Player to play in the current state. WHITE=1, BLACK=-1.
        """
        raise NotImplementedError

    # TODO maybe substitute with step(bool)
    def fake_step(self, action):
        """
        Similar to step(action) in gym.Env, but it doesn't update the current state.
        :param action: object - an action provided by the player
        :return: same as gym.Env.step
        """
        raise NotImplementedError

    def legal_actions(self):
        """
        Returns a list of all legal actions in the current state.
        :return: list - list of all legal actions, a subset of action_space
        """
        raise NotImplementedError

    def winner(self):
        """
        Returns the player that has won the game in the current state. Returns None if the state is not terminal.
        :return: {-1, 0, 1, None} - the player that has won the game: WHITE=1, BLACK=-1, DRAW=0 and None if the game
        has not ended yet
        """
        raise NotImplementedError


class DummyGame(GameEnv):
    """
    Dummy game with a low state space to test GameEnv.

    The game starts with n numbers, from 1 to n, randomly sorted. In every turn a player takes one of the numbers,
    the left-most or the right-most and retires it from the board. The score of each player is the sum of all the
    numbers it has taken and the winner is the player with highest score.

    Example with n=4.

    2 1 4 3
    Player 1 takes the left-most.
    Score: 2-0.

      1 4 3
    Player 2 takes the right-most.
    Score: 2-3.

      1 4
    Player 1 takes the right-most.
    Score: 6-3.

      1
    Player 2 has to take the last number.
    Score: 6-4. Player 1 won.

    Actions: LEFT=0, RIGHT=1.
    """
    board = None
    left = None
    right = None
    scores = None
    _to_play = None

    def __init__(self, n):
        """
        :param n: int - maximum number in the game
        """
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
        """
        Determines if the next move will be the last move of the game (i.e. there is only one number left).
        :return: True if the next move will be the last move of the game, False otherwise
        """
        return self.left == self.right - 1

    def done(self):
        """
        Returns if the game has ended or not.
        :return: True if the game has already ended (i.e. there are no numbers left), False otherwise
        """
        return self.left >= self.right

    def winner(self):
        if not self.done():
            return None
        else:
            return 1 if self.scores[0] > self.scores[1] else \
                -1 if self.scores[1] > self.scores[0] else 0

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
    """
    Wrapper for DummyGame's observations so they can be hashable.
    """

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
