class Callback:
    """Abstract class defining functions used as callback during training.

    This class should not be instantiated directly, but rather it should be
    extended.
    """

    def on_game_end(self, game):
        """Called every time a game ends.

        To make this method work with parallel training delegate the
        modification of internal structures to the join method, as this one
        is called inside each sub-process and join is called in the driver
        process.

        Args:
            game (list[(object, int, numpy.ndarray, int or None)]):
            Representation of a game by a list of tuples (board, turn, pi,
            winner), where board is the state, turn the player to play in
            this state, pi the probability vector of each move and winner is
            the player who has won the game in this state (BLACK, WHITE or
            None).

        Returns:
            object: The result obtained.

        """
        pass

    def join(self, results):
        """Joins the results obtained by on_game_end after all games have
        been played.

        This method should update the internal structures as needed.

        Args:
            results (list[object]): Results obtained by each call of
                on_game_end sorted by the time they were played.

        """
        pass

    def on_update_end(self, actor, info=None):
        """Called after each training of the weights.

        Args:
            actor (tfg.alphaZero.AlphaZero): Actor being trained.
            info (dict): Dict containing some relevant info about the training.
                TODO what

        """
        pass


class PositionCheckerCallback(Callback):

    def __init__(self, encoder=None):
        self._encoder = encoder
        self.states = dict()

    def on_game_end(self, game):
        states = list()
        for state in game:
            board, to_play, _, _ = state
            if self._encoder is not None:
                board = self._encoder(board)
            states.append((board, to_play))
        return states

    def join(self, games):
        for states in games:
            for state in states:
                if state not in self.states:
                    self.states[state] = 1
                else:
                    self.states[state] += 1


class Checkpoint(Callback):

    def __init__(self, prefix="checkpoint", directory='checkpoints', delay=1,
                 verbose=True):
        self.prefix = prefix
        self.directory = directory
        self.delay = delay
        self._counter = 0
        self.verbose = verbose

    def on_update_end(self, actor, info=None):
        self._counter += 1
        if self._counter % self.delay == 0:
            n = str(self._counter // self.delay)
            path = self.directory + '/' + self.prefix + n + '.h5'
            actor.save(path)
            if self.verbose:
                print(f"Checkpoint saved at {path}")
