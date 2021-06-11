import numpy as np
import matplotlib.pyplot as plt
from gym.spaces import Discrete, Box

from tfg.games import GameEnv, WHITE, BLACK


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
        if mode == 'human':
            mapping = {-1: 'O', 0: ' ', 1: 'X'}
            tokens = [[mapping[cell] for cell in row] for row in self.board]
            print("\n-+-+-\n".join(
                ['|'.join([token for token in row]) for row in tokens]
            )+'\n')
        elif mode == 'plt':
            plt.figure()
            plot_board(self.board)
            for i in range(9):
                r, c = self._parse_action(i)
                if self.board[r, c] == 0:
                    plt.text(c + .5, 2 - r + .5, str(i), c='gray', size=18,
                             weight='bold', ha='center', va='center')
            plt.show()
        else:
            raise ValueError('unknown render mode; valid: ' +
                             ', '.join(self.metadata["render.modes"]))

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


def plot_token(i, j, token, color=None, linewidth=4):
    if color is None:
        color = 'k'
    if linewidth is None:
        linewidth = 4

    if token == WHITE:
        x = [j + .2, j + .8]
        y = [2 - i + .2, 2 - i + .8]
        plt.plot(x, y, linewidth=linewidth, c=color)
        plt.plot(x, y[::-1], linewidth=linewidth, c=color)
    elif token == BLACK:
        circle = plt.Circle((j + .5, 2 - i + .5), .35, color=color,
                            linewidth=linewidth, fill=False)
        plt.gca().add_patch(circle)


def plot_board(board, color=None, linewidth=None):
    fig = plt.gcf()
    ax = plt.gca()

    ax.set_axis_off()
    ax.set_aspect('equal', adjustable='box')

    plt.xlim([0, 3])
    plt.ylim([0, 3])

    for i in (1, 2):
        plt.plot([i, i], [0, 3], 'k', linewidth=4)
        plt.plot([0, 3], [i, i], 'k', linewidth=4)

    for i in range(3):
        for j in range(3):
            plot_token(i, j, board[i, j], color, linewidth)

    return fig
