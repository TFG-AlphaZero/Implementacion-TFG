import abc

import gym

WHITE = 1
"""Constant representing first player in a two players game."""
BLACK = -1
"""Constant representing second player in a two players game."""


class GameEnv(gym.Env, abc.ABC):
    """Base class for two player games.

    Extends gym's Env base class to add the capacity to know the result of a
    certain step without actually taking it. It also adds methods to know all
    the legal actions in the current state, a subset of the action space,
    and to know the winner in the current state.

    """
    metadata = {'render.modes': ['human', 'plt']}
    reward_range = (-1, 1)

    @property
    def to_play(self):
        """int: Player to play in the current state. WHITE=1, BLACK=-1."""
        raise NotImplementedError

    def legal_actions(self):
        """Returns a list of all legal actions in the current state.

        Returns:
            list of object: List of all legal actions, a subset of action_space.

        """
        raise NotImplementedError

    def winner(self):
        """Returns the player that has won the game in the current state.
        Returns None if the state is not terminal.

        Returns:
            int or None: Player that has won the game: WHITE=1, BLACK=-1, DRAW=0
                and None if the game has not ended yet.

        """
        raise NotImplementedError
