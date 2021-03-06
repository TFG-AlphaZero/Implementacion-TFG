import copy

import numpy as np
import os
import collections
import time
import random
import tfg.alphaZeroConfig as config

from tfg.strategies import Strategy, MonteCarloTree
from tfg.alphaZeroNN import NeuralNetworkAZ
from tfg.util import play, get_games_per_worker
from tfg.games import BLACK, WHITE
from functools import reduce
from joblib import delayed, Parallel
from threading import Barrier, Thread


class AlphaZero(Strategy):
    """Game strategy implementing AlphaZero algorithm."""

    def __init__(self, env, adapter,
                 c_puct=config.C_PUCT,
                 exploration_noise=config.EXPLORATION_NOISE,
                 mcts_iter=config.MCTS_ITER,
                 mcts_max_time=config.MCTS_MAX_TIME,
                 nn_config=None,
                 gpu=True):
        """All default values are taken from tfg.alphaZeroConfig

        Args:
            env (tfg.games.GameEnv): Game this strategy is for.
            adapter (tfg.alphaZeroAdapters.NeuralNetworkAdapter): Adapter for
                the game given as env.
            c_puct (float): C constant used in the selection policy PUCT
                algorithm.
            exploration_noise ((float, float)): Values used to add Dirichlet
                noise to the first node of MCTS. First number is the noise
                fraction (between 0 and 1), which means how much noise will
                be added. The second number is the alpha of the Dirichlet
                distribution.
            mcts_iter (int): Max iterations of the MCTS algorithm.
            mcts_max_time (float): Max time for the MCTS algorithm.
            nn_config (tfg.alphaZeroConfig.AlphaZeroConfig or dict):
                Wrapper with arguments that will be directly passed to
                tfg.alphaZeroNN.AlphaZeroNN. If output_dim is None,
                env.action_space.n will be used instead.
            gpu (bool): Whether to allow gpu usage for the internal neural
                network or not. Note that Tensorflow should not have been
                imported before creating the actor. Defaults to True.
        """
        if nn_config is None:
            nn_config = config.AlphaZeroConfig()
        elif isinstance(nn_config, dict):
            nn_config = config.AlphaZeroConfig(**nn_config)

        if not gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        self._env = env
        self.c_puct = c_puct
        self.noise_fraction = exploration_noise[0]
        self.noise_alpha = exploration_noise[1]
        self.temperature = 0

        self._nn_input = None
        self._nn_barrier = None
        self._nn_predictions = None
        self._nn_predict = None

        self._thr_num_active = None
        self._thr_actions = None
        self._thr_sync = None

        self._mcts = MonteCarloTree(
            self._env,
            max_iter=mcts_iter,
            max_time=mcts_max_time,
            selection_policy=self._selection_policy,
            value_function=self._value_function,
            best_node_policy=self._best_node_policy,
            reset_tree=True
        )
        
        self.neural_network = NeuralNetworkAZ(
            input_dim=adapter.input_shape,
            output_dim=adapter.output_features,
            **nn_config.__dict__
        )
        self._adapter = adapter

        self.training = True

    @property
    def env(self):
        """tfg.games.GameEnv: Game this strategy is for."""
        return self._env

    def train(self, self_play_times=config.SELF_PLAY_TIMES,
              max_train_time=config.MAX_TRAIN_TIME,
              min_train_error=config.MIN_TRAIN_ERROR,
              max_games_counter=config.MAX_GAMES_COUNTER,
              epochs=config.EPOCHS,
              buffer_size=config.BUFFER_SIZE,
              batch_size=config.BATCH_SIZE,
              temperature=config.TEMPERATURE,
              callbacks=None):
        """Trains the internal neural network via self-play to learn how to
        play the game.

        All parameters default to the values given in tfg.alphaZeroConfig.

        Args:
            self_play_times (int): Number of games played before retraining
                the net.
            max_train_time (float): Maximum time the training can last.
            min_train_error (float): Minimum error below which the training
                will stop.
            max_games_counter (int): Maximum total number of games that can
                be played before stopping training.
            epochs (int): Number of epochs each batch will be trained with.
            buffer_size (int): Max number of states that can be stored before
                training. If this maximum is reached, oldest states will be
                removed when adding new ones.
            batch_size (int): Size of the batch to be sampled of all moves to
                train the network after self_play_times games have been played.
            temperature (int): During first moves of each game actions are
                taken randomly. This parameters sets how many times this will
                be done.
            callbacks (list[tfg.alphaZeroCallbacks.Callback]): Functions that
                will be called after every game and after every set of games to
                join the results.

        """
        def is_done():
            """Training ends if any of the following conditions is met:
                - Training time is over (current_time > max_train_time).
                - Error is lower than threshold
                    (current_error < min_train_error).
                - Max number of played games reached
                    (games_counter > max_games_counter).
            """
            done = False
            if max_train_time is not None:
                done |= (current_time - start_time) > max_train_time
            if min_train_error is not None:
                done |= current_error < min_train_error
            if max_games_counter is not None:
                done |= games_counter >= max_games_counter

            return done

        self.training = True
        self._mcts.reset_tree = True

        # Initialize finishing parameters
        start_time = time.time()
        current_time = start_time
        current_error = float('inf')
        games_counter = 0

        buffer = collections.deque(maxlen=buffer_size)

        while not is_done():
            # Add to buffer the latest played games
            moves, callback_results = self._self_play(
                self_play_times, temperature, callbacks
            )
            buffer.extend(moves)

            # Join callbacks
            if callbacks is not None:
                for callback, results in zip(callbacks, callback_results):
                    callback.join(results)

            # Extract a mini-batch from buffer
            size = min(len(buffer), batch_size)
            mini_batch = random.sample(buffer, size)

            # Separate data from batch
            boards, turns, pies, rewards = zip(*mini_batch)
            train_boards = np.array([
                self._adapter.to_input(board, turn)
                for board, turn in zip(boards, turns)
            ])
            train_pi = np.array(list(pies))
            train_reward = np.array(list(rewards))

            # Train neural network with the data from the mini-batch
            history = self.neural_network.fit(x=train_boards,
                                              y=[train_reward, train_pi],
                                              batch_size=32,
                                              epochs=epochs,
                                              verbose=2,
                                              validation_split=0)
            # Update finishing parameters
            current_error = history.history['loss'][-1]
            current_time = time.time()
            games_counter += self_play_times
            print(f"Games played: {games_counter}")

            info = {
                'error': current_error,
                'time': current_time - start_time,
                'games': games_counter
            }

            if callbacks is not None:
                for callback in callbacks:
                    callback.on_update_end(self, info)

    def _self_play(self, num, temperature, callbacks):
        def make_policy(nodes):
            """Returns the pi vector according to temperature parameter."""
            # Obtain visit vector from children
            visit_vector = np.zeros(self._adapter.output_features)
            for action, node in nodes.items():
                indices = self._adapter.to_indices(action)
                visit_vector[indices] = node.visit_count

            if self.temperature > 0:
                # t = 1 | Exploration
                return visit_vector / visit_vector.sum() 
            else:
                # t -> 0 | Exploitation
                # Vector with all 0s and a 1 in the most visited child
                index = np.unravel_index(np.argmax(visit_vector),
                                         visit_vector.shape)
                pi = np.zeros_like(visit_vector)
                pi[index] = 1
                return pi

        def vale_function(i):
            return lambda node: self._value_function(node, i)

        def thread_run(index, mcts, obs, done):
            def f():
                if not done:
                    action = mcts.move(obs)
                    self._thr_actions[index] = action

                while not self._thr_sync:
                    self._nn_barrier.wait()

            return f

        def multi_predict():
            if self._nn_predict:
                self._nn_predictions = self.neural_network.predict(
                    self._nn_input
                )
                self._nn_predict = False
            else:
                self._thr_sync = True

        callback_results = ([list() for _ in callbacks]
                            if callbacks is not None else [])

        self._nn_input = np.zeros((num, *self._adapter.input_shape))
        self._thr_actions = [None] * num
        self._thr_num_active = num
        self._nn_barrier = Barrier(num, action=multi_predict)

        envs = [copy.deepcopy(self._env) for _ in range(num)]

        mctss = [MonteCarloTree(
            env,
            max_iter=self._mcts.max_iter,
            max_time=self._mcts.max_time,
            selection_policy=self._selection_policy,
            value_function=vale_function(i),
            best_node_policy=self._best_node_policy,
            reset_tree=False
        ) for i, env in enumerate(envs)]

        # Initialize game
        observations = [env.reset() for env in envs]
        dones = [False] * num
        game_data = [list() for _ in range(num)]
        self.temperature = temperature
        s = time.time()

        # Loop until all games end
        while self._thr_num_active > 0:
            self._nn_predict = False
            self._thr_sync = False

            # Launch threads to choose moves from MCTS
            threads = [Thread(target=thread_run(i, mcts, o, d))
                       for i, (mcts, o, d)
                       in enumerate(zip(mctss, observations, dones))]

            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            # Threads have been synchronized
            actions = self._thr_actions

            # Update temperature parameter
            self.temperature = max(0, self.temperature - 1)

            # Make all moves
            for i in range(num):
                if dones[i]:
                    continue
                # Calculate Pi vector
                pi = make_policy(mctss[i].stats['actions'])
                # Store move data: (board, turn, pi)
                game_data[i].append((observations[i], envs[i].to_play, pi))

                # Perform move
                observation, _, done, _ = envs[i].step(actions[i])
                observations[i] = observation
                dones[i] = done
                # Update MCTS (used to recycle tree for next move)
                mctss[i].update(actions[i])

                if done:
                    self._thr_num_active -= 1
                    n = self._adapter.output_features
                    pi = np.full(n, 1 / n)
                    game_data[i].append((observation, envs[i].to_play, pi))
                    # If game is over: exit loop
                    print(f"game finished in {time.time() - s}")

        # Store winner in all states gathered
        for i, (data, env) in enumerate(zip(game_data, envs)):
            winner = env.winner()
            for j in range(len(data)):
                to_play = data[j][1]
                data[j] += (winner * to_play,)

        # Flatten all results
        game_data = [d for data in game_data for d in data]

        # Add callback results
        if callbacks is not None:
            for callback, result in zip(callbacks, callback_results):
                result.append(callback.on_game_end(game_data))

        return game_data, callback_results

    def move(self, observation):
        self.temperature = 0
        self.training = False
        self._mcts.reset_tree = False
        return self._mcts.move(observation)

    def update(self, action):
        self._mcts.update(action)

    def set_max_iter(self, max_iter):
        """Sets max_iter attribute of the internal MCTS.

        Args:
            max_iter (int): Maximum number of iterations of MCTS.

        """
        self._mcts.max_iter = max_iter

    def set_max_time(self, max_time):
        """Sets max_time attribute of the internal MCTS.

        Args:
            max_time (float): Maximum time the MCTS algorithm can run.

        """
        self._mcts.max_time = max_time

    def save(self, path):
        """Saves the internal neural network model in the given path.

        Args:
            path (str): Where to save the model.

        """
        self.neural_network.save_model(path)

    def load(self, path):
        """Restores the internal neural network model from the given path.

        Args:
            path (str): Where to load the model from.

        """
        self.neural_network.load_model(path)

    def get_weights(self):
        """Retrieves the weights of the internal neural network model.

        Returns:
            list[numpy.ndarray]: Weights of the internal neural network.

        """
        return self.neural_network.model.get_weights()

    def set_weights(self, weights):
        """Sets the weights of the internal neural network from Numpy arrays.

        Args:
            weights ({__len__}):

        """
        self.neural_network.model.set_weights(weights)

    def _value_function(self, node, index=None):
        # Convert node to network input format
        nn_input = np.array([
            self._adapter.to_input(node.observation, node.to_play)
        ])

        if index is not None:
            # Synchronize
            self._nn_input[index] = nn_input
            self._nn_predict = True
            # Wait until all threads are ready
            self._nn_barrier.wait()
            # Barrier predicts before exiting
            reward = self._nn_predictions[0][index, 0]
            probabilities = self._nn_predictions[1][index]
        else:
            # Predict Node with Neural Network
            predictions = self.neural_network.predict(nn_input)
            # Extract output data
            reward = predictions[0][0, 0]
            probabilities = predictions[1][0]

        # Obtain legal actions
        legal_actions = list(node.children.keys())
        legal_indices = [self._adapter.to_indices(action)
                         for action in legal_actions]
        # Expand indices if necessary
        mask_indices = (legal_indices if len(probabilities.shape) == 1 else
                        tuple(zip(*legal_indices)))

        # Obtain only legal probabilities and interpolate them
        mask = np.zeros_like(probabilities, dtype=bool)
        mask[mask_indices] = True
        if (probabilities[mask] == 0).all():
            # All probabilities may be zero
            probabilities[mask] = 1 / mask.sum()
        else:
            probabilities[mask] /= probabilities[mask].sum()
        probabilities[~mask] = 0

        # Add exploration noise if node is root
        if self.training and node.root:
            alpha = np.full(len(legal_actions), self.noise_alpha)
            noise, = np.random.dirichlet(alpha, size=1)
            probabilities[mask] = (
                    (1 - self.noise_fraction) * probabilities[mask]
                    + self.noise_fraction * noise
            )

        # Assign probabilities to children
        for action, indices in zip(legal_actions, legal_indices):
            node.children[action].probability = probabilities[indices]

        return reward

    def _best_node_policy(self, children):
        """Function representing the best node policy from AlphaZero
        used by the Monte Carlo Tree Search

        Formula:
        pi(a|s) = N(s,a)^(1/t) / Sum_b N(s,b)^(1/t)
        where t is a temperature parameter and b denotes all available
        actions at state s.

        - t = 1 means a high level of exploration.
        - t -> 0 it means a low exploration.
        """
        visit_vector = np.zeros(len(children))
        for i in range(len(children)):
            visit_vector[i] = children[i].visit_count

        if self.temperature > 0:
            # t = 1 | Exploration | Sample from categorical distribution
            sample = random.random()
            distribution = np.cumsum(visit_vector / visit_vector.sum())
            return np.argmax(distribution > sample)
        else:
            # t -> 0 | Exploitation | Node with highest visit count
            return np.argmax(visit_vector)

    def _selection_policy(self, root, children):
        """Function representing the Q + U formula. (Deep RL)
        
        In the step of select, the action is selected by the formula:
        a = argmax(Q(s,a) + U(s,a)) where
        - Q(s,a) = W/N encourages the exploitation
        - U(s,a) = c_puct * P(s,a) * sqrt(Sum_b N(s,b)) / 1 + N(s,a) encourages
          the exploration
        
        and c_puct is a parameter determining the exploration scale.
        """
        Q = np.array([child.value for child in children])
        
        visits = np.array([child.visit_count for child in children])
        probabilities = np.array([child.probability for child in children])

        U = self.c_puct * probabilities * (
                np.sqrt(root.visit_count) / (1 + visits)
        )

        res = Q + U
        return np.argmax(res)


def create_alphazero(game, adapter,
                     initial_weights=None,
                     max_workers=None,
                     buffer_size=config.BUFFER_SIZE,
                     self_play_times=config.SELF_PLAY_TIMES,
                     max_train_time=config.MAX_TRAIN_TIME,
                     min_train_error=config.MIN_TRAIN_ERROR,
                     max_games_counter=config.MAX_GAMES_COUNTER,
                     batch_size=config.BATCH_SIZE,
                     epochs=config.EPOCHS,
                     temperature=config.TEMPERATURE,
                     callbacks=None,
                     *args, **kwargs):
    """Creates and trains a new instance of AlphaZero.

    This function allow parallel training whereas AlphaZero.train does not.

    Args:
        game (tfg.games.GameEnv): Game AlphaZero is for.
        adapter (tfg.alphaZeroAdapters.NeuralNetworkAdapter): Adapter for the
            game given as env.
        initial_weights (str): File to load initial weights from.
        max_workers(int): Number of processes that will be used.
        buffer_size (int): Max number of states that can be stored before
            training. If this maximum is reached, oldest states will be
            removed when adding new ones.
        self_play_times (int): Number of games played before retraining
            the net.
        max_train_time (float): Maximum time the training can last.
        min_train_error (float): Minimum error bellow which the training
            will stop.
        max_games_counter (int): Maximum total number of games that can
            be played before stopping training.
        epochs (int): Number of epochs each batch will be trained with.
        batch_size (int): Size of the batch to be sampled of all moves to
            train the network after self_play_times games have been played.
        temperature (int): During first moves of each game actions are
            taken randomly. This parameters sets how many times this will
            be done.
        callbacks (list[tfg.alphaZeroCallbacks.Callback]): Functions that
            will be called after every game and after every set of games to
            join the results.
        *args: Other arguments passed to AlphaZero's constructor starting
            after adapter.
        **kwargs: Other keyword arguments passed to AlphaZero's constructor.

    """

    if max_workers is None:
        actor = AlphaZero(game, adapter, *args, **kwargs)
        actor.train(self_play_times, max_train_time, min_train_error,
                    max_games_counter, epochs, batch_size, temperature)
        return actor

    import ray

    if not ray.is_initialized():
        ray.init(log_to_driver=False)

    def is_done():
        done = False
        if max_train_time is not None:
            done |= (current_time - start_time) > max_train_time
        if min_train_error is not None:
            done |= current_error < min_train_error
        if max_games_counter is not None:
            done |= games_counter >= max_games_counter

        return done

    AZ = ray.remote(AlphaZero)
    azs = [AZ.remote(game, adapter, *args, gpu=False, **kwargs)
           for _ in range(max_workers)]
    actor = AlphaZero(game, adapter, *args, **kwargs)

    if initial_weights is not None:
        actor.load(initial_weights)

    start_time = time.time()
    current_time = start_time
    current_error = float('inf')
    games_counter = 0

    buffer = collections.deque(maxlen=buffer_size)

    n_games = get_games_per_worker(self_play_times, max_workers)

    while not is_done():
        weights = actor.get_weights()
        ray.get([az.set_weights.remote(weights) for az in azs])

        futures = [az._self_play.remote(g, temperature, callbacks)
                   for az, g in zip(azs, n_games)]
        moves, callback_results = zip(*ray.get(futures))

        # Join callbacks
        if callbacks is not None:
            # For each worker
            for callback_results_ in callback_results:
                for callback, results in zip(callbacks, callback_results_):
                    callback.join(results)

        for m in moves:
            buffer.extend(m)

        size = min(len(buffer), batch_size)
        mini_batch = random.sample(buffer, size)

        boards, turns, pies, rewards = zip(*mini_batch)
        train_boards = np.array([
            adapter.to_input(board, turn) for board, turn in zip(boards, turns)
        ])
        train_pi = np.array(list(pies))
        train_reward = np.array(list(rewards))

        history = actor.neural_network.fit(x=train_boards,
                                           y=[train_reward, train_pi],
                                           batch_size=32,
                                           epochs=epochs,
                                           verbose=2,
                                           validation_split=0)

        current_error = history.history['loss'][-1]
        current_time = time.time()
        games_counter += self_play_times

        info = {
            'error': current_error,
            'time': current_time - start_time,
            'games': games_counter,
            'remote_actors': azs
        }

        print(f"Games played: {games_counter}")

        if callbacks is not None:
            for callback in callbacks:
                callback.on_update_end(actor, info)

    return actor


def parallel_play(game, adapter, rival, weights_file=None, color=WHITE,
                  games=100, max_workers=4, *args, **kwargs):
    """Utility function that allows to play games with AlphaZero using multiple
    processors.

    Args:
        game (tfg.games.GameEnv): Game to be played.
        adapter (tfg.alphaZeroAdapters.NeuralNetworkAdapter): Adapter for the
            game given as env.
        rival (tfg.strategies.Strategy): AlphaZero's opponent.
        weights_file (str): File to load weights from.
        color (int or str): AlphaZero's color. Either BLACK (-1) or WHITE (1),
            or 'black' or 'white'. Defaults to WHITE.
        games (int): Number of games to be played. Defaults to 100.
        max_workers (int): Number of processes that will be used.
        *args: Other arguments passed to AlphaZero's constructor starting
            after adapter.
        **kwargs: Other keyword arguments passed to AlphaZero's constructor.

    Returns:
        ((int, int, int)): Games won by white, drawn and won by black.

    """
    def play_(n):
        az = AlphaZero(game, adapter, *args, gpu=False, **kwargs)
        if weights_file is not None:
            az.load(weights_file)
        if color in (WHITE, 'white'):
            return play(game, az, rival, games=n)
        elif color in (BLACK, 'black'):
            return play(game, rival, az, games=n)
        else:
            raise ValueError(f"invalid color: {color} "
                             f"(expected 'white' (1) or 'black' (-1))")

    n_games = get_games_per_worker(games, max_workers)

    results = Parallel(max_workers)(delayed(play_)(g) for g in n_games)
    return tuple(reduce(lambda acc, x: map(sum, zip(acc, x)), results))
