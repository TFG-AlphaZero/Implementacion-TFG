import sys
sys.path.insert(0, '/Documents/Juan Carlos/Estudios/Universidad/5º Carrera/TFG Informatica/ImplementacionTFG')

import numpy as np
import copy
import collections
import time
import itertools
import random
import tfg.alphaZeroConfig as config

from tfg.strategies import Strategy, MonteCarloTree
from tfg.alphaZeroNN import NeuralNetworkAZ
from joblib import Parallel, delayed

# TODO gpu is disabled here by force because training wouldn't work otherwise
#   it may be a good idea disabling only if necessary
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class AlphaZero(Strategy):
    """Game strategy implementing AlphaZero algorithm."""

    # TODO add custom action/observation encoders/decoders
    def __init__(self, env,
                 c_puct=config.C_PUCT,
                 mcts_times=config.MCTS_TIMES,
                 mcts_max_time=config.MCTS_MAX_TIME,
                 t_equals_one=config.T_EQUALS_ONE,
                 buffer_size=config.BUFFER_SIZE,
                 nn_config=None):
        """All default values are taken from tfg.alphaZeroConfig

        Args:
            env (tfg.games.GameEnv): Game this strategy is for.
            c_puct (float): C constant used in the selection policy PUCT
                algorithm.
            mcts_times (int): Max iterations of the MCTS algorithm.
            mcts_max_time (float): Max time for the MCTS algorithm.
            buffer_size (int): Max number of states that can be stored before
                training. If this maximum is reached, oldest states will be
                removed when adding new ones.
            nn_config (tfg.alphaZeroConfig.AlphaZeroConfig or dict):
                Wrapper with arguments that will be directly passed to
                tfg.alphaZeroNN.AlphaZeroNN. If output_dim is None,
                env.action_space.n will be used instead.
        """
        if nn_config is None:
            nn_config = config.AlphaZeroConfig()
        elif isinstance(nn_config, dict):
            nn_config = config.AlphaZeroConfig(**nn_config)

        self._env = env
        self.mcts_kwargs = dict(
            max_iter=mcts_times,
            max_time=mcts_max_time,
            selection_policy=QPlusU(c_puct),
            value_function=self._value_function,
            best_node_policy=self._best_node_policy,
            reset_tree=False
        )

        # FIXME we shouldn't use config here
        self.input_dim = self._env.observation_space.shape + config.INPUT_LAYERS

        self._mcts = MonteCarloTree(self._env, **self.mcts_kwargs)

        self._buffer = collections.deque(maxlen=buffer_size)

        self.temperature = 0

        if nn_config.output_dim is None:
            # Try using same output dim as action space size
            nn_config.output_dim = self._env.action_space.n
        
        self.neural_network = NeuralNetworkAZ(
            input_dim=self.input_dim,
            **nn_config.__dict__
        )

    @property
    def env(self):
        """tfg.games.GameEnv: Game this strategy is for."""
        return self._env

    def train(self, self_play_times=config.SELF_PLAY_TIMES,
              max_train_time=config.MAX_TRAIN_TIME,
              max_train_error=config.MAX_TRAIN_ERROR,
              max_games_counter=config.MAX_GAMES_COUNTER,
              max_workers=config.MAX_WORKERS,
              epochs=config.EPOCHS,
              batch_size=config.BATCH_SIZE,
              t_equals_one=config.T_EQUALS_ONE):
        """
        TODO
        """
        def is_done():
            """
            Training ends if any of the following conditions is met:
                - Training time is over (current_time > max_train_time)
                - Error is lower than threshold (current_error < max_train_error)
                - Max number of played games reached  (games_counter > max_games_counter)
            """
            done = False
            if max_train_time is not None:
                done |= (current_time - start_time) > max_train_time
            if max_train_error is not None:
                done |= current_error < max_train_error
            if max_games_counter is not None:
                done |= games_counter > max_games_counter

            return done

        #Initialize finishing parameters
        start_time = time.time()
        current_time = start_time
        current_error = float('inf')
        games_counter = 0

        while not is_done():
            #Add to buffer the latest played games
            self._buffer.extend(
                self._self_play(self_play_times, max_workers, t_equals_one)
            )

            #Extract a mini-batch from buffer
            size = min(len(self._buffer), batch_size)
            mini_batch = random.sample(self._buffer, size)
            
            #Separate data from batch
            boards, turns, pies, rewards = zip(*mini_batch)
            train_board = np.array([
                self._convert_to_network_input(boards[i], turns[i]) for i in range(len(boards))]
            )
            train_pi = np.array(list(pies))
            train_reward = np.array(list(rewards))

            #Train neural network with the data from the mini-batch
            history = self.neural_network.fit(x=train_board, y=[train_reward, train_pi],
                                              batch_size=32,
                                              epochs=epochs,
                                              verbose=2,
                                              validation_split=0)
            #Update finishing parameters
            current_error = history.history['loss'][-1]
            current_time = time.time()
            games_counter += self_play_times

    def _self_play(self, games, max_workers, t_equals_one):
        def make_policy(env, nodes, temperature):
            """Returns the pi vector according to temperature parameter
            """
            #Obtain visit vector from children
            visit_vector = np.zeros(env.action_space.n)
            for action, node in nodes.items():
                # TODO We are assuming action is an int or a tuple,
                #  check when generalizing
                visit_vector[action] = node.visit_count

            if temperature > 0:
                # t = 1 | Exploration
                return visit_vector / visit_vector.sum() 
            else:
                # t->0 | Explotation
                #Vector with all 0s and a 1 in the most visisted child
                index = np.argmax(visit_vector)
                pi = np.zeros(visit_vector.size)
                pi[index] = 1
                return pi

        def _self_play_(num, env, mcts):
            game_buffer = []

            #Play num games
            for _ in range(num):
                #Initialize game
                observation = env.reset()
                game_data = []
                self.temperature = t_equals_one
                s = time.time()

                #Loop until game ends
                while True:
                    #Choose move from MCTS
                    action = mcts.move(observation)
                    
                    #Calculate Pi vector
                    pi = make_policy(env, mcts.stats['actions'], self.temperature) 
                    #Update temperature parameter
                    self.temperature = max(0, self.temperature - 1)                    
                    #Store move data: (board, turn, pi)
                    game_data.append((observation, env.to_play, pi))
                    
                    #Perform move
                    observation, _, done, _ = env.step(action)
                    #Update MCTS (used to recycle tree for next move) 
                    mcts.update(action)
                    
                    if done:
                        #If game is over: exit loop
                        print(f"game finished in {time.time() - s}")
                        break

                #Store winner in all states gathered
                #Change winner perspective whether white or black has won
                perspective = 1 if game_data[-1][1] == 1 else -1
                #Iterate list in reversed order (and modifying it)
                for i in range(len(game_data) - 1, -1, -1):
                    # TODO turns may not switch every time,
                    #  check when generalizing
                    #Store winner in move data according to turn perspective
                    game_data[i] += (perspective * env.winner(),)
                    perspective *= -1

                #Add game states to buffer: (board, turn, pi, winner)
                game_buffer.extend(game_data)

            return game_buffer

        # TODO mover esto a la función anterior
        if max_workers is None:
            return _self_play_(games, self._env, self._mcts)

        d_games = games // max_workers
        r_games = games % max_workers
        n_games = [d_games] * max_workers
        if r_games != 0:
            for i in range(r_games):
                n_games[i] += 1

        envs = [copy.deepcopy(self._env) for _ in range(max_workers)]
        mctss = [MonteCarloTree(env, **self.mcts_kwargs) for env in envs]
        args = zip(n_games, envs, mctss)

        bufs = Parallel(max_workers, backend='threading')(
            delayed(_self_play_)(g, env, mcts) for g, env, mcts in args
        )
        return list(itertools.chain.from_iterable(bufs))

    def move(self, observation):
        self.temperature = 0
        return self._mcts.move(observation)

    def update(self, action):
        self._mcts.update(action)

    def save(self, path):
        self.neural_network.save_model(path)

    def load(self, path):
        self.neural_network.load_model(path)

    def _value_function(self, node):
        #Convert node to network input format
        nn_input = np.array([self._convert_to_network_input(node.observation, node.to_play)])
        
        #Predict Node with Neural Network
        predictions = self.neural_network.predict(nn_input)
        
        #Extract output data
        reward = predictions[0][0][0]
        probabilities = predictions[1][0]
        
        #Assign probabilities to children
        for i, child in node.children.items():
            child.probability = probabilities[i]

        return reward

    def _best_node_policy(self, children) :
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
            # t -> 0 | Explotation | Node with highest visit count
            return np.argmax(visit_vector)

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

        #All 1 if black to play, all 0 if white to play
        player = 0 if to_play == 1 else 1
        turn = np.full(board.shape, player)

        #Join layers together with channel_last
        input = np.stack((black, white, turn), axis = 2)

        return input

    def get_weights(self):
        return self.neural_network.model.get_weights()

    def set_weights(self, weights):
        self.neural_network.model.set_weights(weights)


class QPlusU:
    """Class representing the Q + U formula. (Deep RL)
    
    In the step of select, the action is selected by the formula 
    a = argmax(Q(s,a) + U(s,a)) where Q(s,a) = W/N encourages the exploitation
    and U(s,a) = c_puct * P(s,a) * sqrt(Sum_b N(s,b)) / 1 + N(s,a) encourages
    the exploration.
    
    c_puct is a parameter determining the exploration scale (5 for AlphaZero).

    It is a functional class: (MonteCarloTreeNode, [MonteCarloTreeNode]) -> int.
    """

    def __init__(self, c_puct):
        """

        Args:
            c_puct (float): Exploration scale constant.

        """
        self.c_puct = c_puct

    def __call__(self, root, children):
        Q = np.array([child.value for child in children])
        
        visits = np.array([child.visit_count for child in children])
        probabilities = np.array([child.probability for child in children])

        U = self.c_puct * probabilities * (
                np.sqrt(root.visit_count) / (1 + visits)
        )

        res = Q + U
        return np.argmax(res)


def create_alphazero(game, max_workers=None,
                     self_play_times=config.SELF_PLAY_TIMES,
                     max_train_time=config.MAX_TRAIN_TIME,
                     max_train_error=config.MAX_TRAIN_ERROR,
                     max_games_counter=config.MAX_GAMES_COUNTER,
                     batch_size=config.BATCH_SIZE,
                     buffer_size=config.BUFFER_SIZE,
                     epochs=config.EPOCHS,
                     t_equals_one=config.T_EQUALS_ONE,
                     *args, **kwargs):

    if max_workers is None:
        raise NotImplementedError("not implemented yet")

    import ray

    if not ray.is_initialized():
        ray.init(log_to_driver=False)

    def is_done():
        done = False
        if max_train_time is not None:
            done |= (current_time - start_time) > max_train_time
        if max_train_error is not None:
            done |= current_error < max_train_error
        # if max_games_counter is not None:
        #     done |= games_counter > max_games_counter

        return done

    AZ = ray.remote(AlphaZero)
    azs = [AZ.remote(game, *args, **kwargs) for _ in range(max_workers)]
    actor = AlphaZero(game, *args, **kwargs)

    start_time = time.time()
    current_time = start_time
    current_error = float('inf')

    buffer = collections.deque(maxlen=buffer_size)

    d_games = self_play_times // max_workers
    r_games = self_play_times % max_workers
    n_games = [d_games] * max_workers
    if r_games != 0:
        for i in range(r_games):
            n_games[i] += 1

    while not is_done():
        futures = [az._self_play.remote(g, None, t_equals_one)
                   for az, g in zip(azs, n_games)]
        moves = ray.get(futures)
        for m in moves:
            buffer.extend(m)

        size = min(len(buffer), batch_size)
        mini_batch = random.sample(buffer, size)

        boards, turns, pies, rewards = zip(*mini_batch)
        x_train = np.array([
            actor._convert_to_network_input(board, turn)
            for board, turn in zip(boards, turns)
        ])
        train_prob = np.array(list(pies))
        train_reward = np.array(list(rewards))

        history = actor.neural_network.fit(x=x_train,
                                           y=[train_reward, train_prob],
                                           batch_size=32,
                                           epochs=epochs,
                                           verbose=2,
                                           validation_split=0)

        weights = actor.get_weights()
        futures = [az.set_weights.remote(weights) for az in azs]
        ray.get(futures)

        current_error = history.history['loss'][-1]
        current_time = time.time()

    return actor
