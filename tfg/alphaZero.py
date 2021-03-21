import sys
sys.path.insert(0, '/Documents/Juan Carlos/Estudios/Universidad/5º Carrera/TFG Informatica/ImplementacionTFG')

import numpy as np
import os
import collections
import time
import random
import tfg.alphaZeroConfig as config

from tfg.strategies import Strategy, MonteCarloTree
from tfg.alphaZeroNN import NeuralNetworkAZ


class AlphaZero(Strategy):
    """Game strategy implementing AlphaZero algorithm."""

    # TODO add custom action/observation encoders/decoders
    def __init__(self, env,
                 c_puct=config.C_PUCT,
                 exploration_noise = config.EXPLORATION_NOISE,
                 mcts_iter=config.MCTS_ITER,
                 mcts_max_time=config.MCTS_MAX_TIME,
                 buffer_size=config.BUFFER_SIZE,
                 nn_config=None,
                 gpu=True):
        """All default values are taken from tfg.alphaZeroConfig

        Args:
            env (tfg.games.GameEnv): Game this strategy is for.
            c_puct (float): C constant used in the selection policy PUCT
                algorithm.
            mcts_iter (int): Max iterations of the MCTS algorithm.
            mcts_max_time (float): Max time for the MCTS algorithm.
            buffer_size (int): Max number of states that can be stored before
                training. If this maximum is reached, oldest states will be
                removed when adding new ones.
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
        # FIXME we shouldn't use config here
        self.input_dim = self._env.observation_space.shape + config.INPUT_LAYERS
        self.c_puct = c_puct
        self.noise_fraction = exploration_noise[0]
        self.noise_alpha = exploration_noise[1]
        self.temperature = 0
        self._buffer = collections.deque(maxlen=buffer_size)

        self._mcts = MonteCarloTree(
            self._env,
            max_iter=mcts_iter,
            max_time=mcts_max_time,
            selection_policy=self._selection_policy,
            value_function=self._value_function,
            best_node_policy=self._best_node_policy,
            reset_tree=False
        )

        if nn_config.output_dim is None:
            # Try using same output dim as action space size
            nn_config.output_dim = self._env.action_space.n
        
        self.neural_network = NeuralNetworkAZ(
            input_dim=self.input_dim,
            **nn_config.__dict__
        )

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

        # Initialize finishing parameters
        start_time = time.time()
        current_time = start_time
        current_error = float('inf')
        games_counter = 0

        while not is_done():
            # Add to buffer the latest played games
            moves, callback_results = self._self_play(
                self_play_times, temperature, callbacks
            )
            self._buffer.extend(moves)

            # Join callbacks
            if callbacks is not None:
                for callback, results in zip(callbacks, callback_results):
                    callback.join(results)

            # Extract a mini-batch from buffer
            size = min(len(self._buffer), batch_size)
            mini_batch = random.sample(self._buffer, size)

            # Separate data from batch
            boards, turns, pies, rewards = zip(*mini_batch)
            train_board = np.array([
                self._convert_to_network_input(board, turn)
                for board, turn in zip(boards, turns)]
            )
            train_pi = np.array(list(pies))
            train_reward = np.array(list(rewards))

            # Train neural network with the data from the mini-batch
            history = self.neural_network.fit(x=train_board,
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

            if callbacks is not None:
                for callback in callbacks:
                    callback.on_update_end(self)

    def _self_play(self, num, temperature, callbacks):
        def make_policy(env, nodes):
            """Returns the pi vector according to temperature parameter."""
            # Obtain visit vector from children
            visit_vector = np.zeros(env.action_space.n)
            for action, node in nodes.items():
                # TODO We are assuming action is an int or a tuple,
                #  check when generalizing
                visit_vector[action] = node.visit_count

            if self.temperature > 0:
                # t = 1 | Exploration
                return visit_vector / visit_vector.sum() 
            else:
                # t -> 0 | Exploitation
                # Vector with all 0s and a 1 in the most visited child
                index = np.argmax(visit_vector)
                pi = np.zeros(visit_vector.size)
                pi[index] = 1
                return pi

        game_buffer = []
        callback_results = ([list() for _ in callbacks]
                            if callbacks is not None else [])

        # Play num games
        for _ in range(num):
            # Initialize game
            observation = self._env.reset()
            game_data = []
            self.temperature = temperature
            s = time.time()

            # Loop until game ends
            while True:
                # Choose move from MCTS
                action = self._mcts.move(observation)

                # Calculate Pi vector
                pi = make_policy(self._env, self._mcts.stats['actions'])
                # Update temperature parameter
                self.temperature = max(0, self.temperature - 1)
                # Store move data: (board, turn, pi)
                game_data.append((observation, self._env.to_play, pi))

                # Perform move
                observation, _, done, _ = self._env.step(action)
                # Update MCTS (used to recycle tree for next move)
                self._mcts.update(action)

                if done:
                    # If game is over: exit loop
                    print(f"game finished in {time.time() - s}")
                    break

            # Store winner in all states gathered
            for i in range(len(game_data)):
                game_data[i] += (self._env.winner(),)

            # Add game states to buffer: (board, turn, pi, winner)
            game_buffer.extend(game_data)

            # Add callback results
            if callbacks is not None:
                for callback, result in zip(callbacks, callback_results):
                    result.append(callback.on_game_end(game_data))

        return game_buffer, callback_results

    def move(self, observation):
        self.temperature = 0
        self.training = False
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

    def _value_function(self, node):
        # Convert node to network input format
        nn_input = np.array([
            self._convert_to_network_input(node.observation, node.to_play)
        ])
        
        # Predict Node with Neural Network
        predictions = self.neural_network.predict(nn_input)
        
        # Extract output data
        reward = predictions[0][0][0]
        probabilities = predictions[1][0]
        
        # Add exploration noise if node is root 
        if self.training and node.root:
            alpha = np.full(len(probabilities), self.noise_alpha)
            noise = np.random.dirichlet(alpha, size=1)
            probabilities = ((1 - self.noise_fraction) * probabilities
                             + self.noise_fraction * noise[0])

        # Obtain legal actions
        legal_actions = list(node.children.keys())

        # Obtain only legal probabilities and interpolate them
        mask = np.zeros_like(probabilities, dtype=bool)
        mask[legal_actions] = True
        probabilities[mask] /= probabilities[mask].sum()
        probabilities[~mask] = 0

        # Create dictionary to assign probabilities properly as
        # children.items() may not have same order than legal_actions
        # prob_dic = dict(zip(legal_actions, probabilities))

        # Assign probabilities to children
        for action in legal_actions:
            node.children[action].probability = probabilities[action]

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

    @staticmethod
    def _convert_to_network_input(board, to_play):
        """Converts from raw board and turn format
        into neural network format
        """
        black = board.copy()
        black[black == 1] = 0
        black[black == -1] = 1

        white = board.copy()
        white[white == -1] = 0

        # All 1 if black to play, all 0 if white to play
        player = 0 if to_play == 1 else 1
        turn = np.full(board.shape, player)

        # Join layers together with channel_last
        input = np.stack((black, white, turn), axis=2)

        return input


def create_alphazero(game, max_workers=None,
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
        max_workers(int): Number of processes that will be used.
        buffer_size (int): Max number of states that can be stored before
            training. If this maximum is reached, oldest states will be
            removed when adding new ones. This is used inside this function
            and the actor will be created using this parameter as well.
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
            after game.
        **kwargs: Other keyword arguments passed to AlphaZero's constructor.

    """

    if max_workers is None:
        actor = AlphaZero(game, *args, buffer_size=buffer_size, **kwargs)
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
    azs = [AZ.remote(game, *args, buffer_size=buffer_size, gpu=False, **kwargs)
           for _ in range(max_workers)]
    actor = AlphaZero(game, *args, buffer_size=buffer_size, **kwargs)

    start_time = time.time()
    current_time = start_time
    current_error = float('inf')
    games_counter = 0

    buffer = collections.deque(maxlen=buffer_size)

    d_games = self_play_times // max_workers
    r_games = self_play_times % max_workers
    n_games = [d_games] * max_workers
    if r_games != 0:
        for i in range(r_games):
            n_games[i] += 1

    while not is_done():
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
        train_board = np.array([
            actor._convert_to_network_input(board, turn)
            for board, turn in zip(boards, turns)
        ])
        train_pi = np.array(list(pies))
        train_reward = np.array(list(rewards))

        history = actor.neural_network.fit(x=train_board,
                                           y=[train_reward, train_pi],
                                           batch_size=32,
                                           epochs=epochs,
                                           verbose=2,
                                           validation_split=0)

        weights = actor.get_weights()
        if callbacks is not None:
            for callback in callbacks:
                callback.on_update_end(actor)
        futures = [az.set_weights.remote(weights) for az in azs]
        ray.get(futures)

        current_error = history.history['loss'][-1]
        current_time = time.time()
        games_counter += self_play_times
        print(f"Games played: {games_counter}")

    return actor
