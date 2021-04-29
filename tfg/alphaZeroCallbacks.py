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

    def on_update_end(self, actor, info):
        """Called after each training of the weights.

        Args:
            actor (tfg.alphaZero.AlphaZero): Actor being trained.
            info (dict): Dict containing some relevant info about the training.
                This dict contains 'error', the current error; 'time',
                elapsed training time; 'games', number of games played;
                'remote_actors' (optional), a list of pointers to the remote
                ray actors if training in parallel.

        """
        pass


class Checkpoint(Callback):
    """Callback that saves a checkpoint of the model after updating weights."""

    def __init__(self, prefix='checkpoint', directory='checkpoints', delay=1,
                 verbose=True):
        """

        Args:
            prefix (str): Filename prefix. Defaults to 'checkpoint'.
            directory (str): Where to save the models. Defaults to
                'checkpoints'.
            delay (int): Number of weight updates between one checkpoint and
                the next. Defaults to 1.
            verbose (bool): Whether to print the filename after saving or
                not. Defaults to True.

        """
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


class ParamScheduler(Callback):
    """Callback that allows the modification of some hyperparameters."""

    def __init__(self,
                 mtcs_iter_schedule=None,
                 lr_schedule=None,
                 verbose=True):
        """

        Args:
            mtcs_iter_schedule (dict[int, int]): Mapping from number of
                games played to number of MCTS iterations. Optional.
            lr_schedule (dict[int, float]): Mapping from number of
                games played to neural network's learning rate. Optional.
            verbose (bool): Whether to print the value set after setting it.
                Defaults to True.

        """

        self.mcts_iter_schedule = self._convert_schedule(mtcs_iter_schedule)
        self.lr_schedule = self._convert_schedule(lr_schedule)

        self.verbose = verbose

    def on_update_end(self, actor, info):
        self._update_max_iter(actor, info)
        self._update_lr(actor, info)

    def _update_max_iter(self, actor, info):
        if not self.mcts_iter_schedule:
            return
        games = info['games']
        max_iter = self._get_next(self.mcts_iter_schedule, games)
        if max_iter is None:
            return
        actor.set_max_iter(max_iter)
        if 'remote_actors' in info:
            import ray
            ray.get([az.set_max_iter.remote(max_iter)
                     for az in info['remote_actors']])
        if self.verbose:
            print(f"max_iter set: {max_iter}")

    def _update_lr(self, actor, info):
        if not self.lr_schedule:
            return
        games = info['games']
        lr = self._get_next(self.lr_schedule, games)
        if lr is None:
            return
        actor.neural_network.model.optimizer.lr.assign(lr)
        if self.verbose:
            print(f"learning_rate set: {lr}")

    @staticmethod
    def _get_next(schedule, games):
        result = None
        while schedule and games >= schedule[0][0]:
            result = schedule.pop(0)[1]
        return result

    @staticmethod
    def _convert_schedule(schedule):
        if schedule is None:
            return None
        if isinstance(schedule, dict):
            schedule = schedule.items()
        return list(sorted(schedule))


class GameStore(Callback):
    """Callback that stores all games played during training.

    Now it might be broken as several games are played at the same time
    during self-play and they won't get appended properly.
    """

    def __init__(self):
        self.games = []

    def on_game_end(self, game):
        boards = list()
        for state in game:
            board = state[0]
            boards.append(board)
        return boards

    def join(self, games):
        self.games.extend(games)

    def print(self, index):
        print(*self.games[index], sep='\n\n')
