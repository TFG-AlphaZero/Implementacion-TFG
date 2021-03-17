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
                training. If this maximum is reached, old states will be
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
            best_node_policy='robust',
            reset_tree=False
        )

        # FIXME we shouldn't use config here
        self.input_dim = self._env.observation_space.shape + config.INPUT_LAYERS

        self._mcts = MonteCarloTree(self._env, **self.mcts_kwargs)

        self._buffer = collections.deque(maxlen=buffer_size)

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
              max_workers=config.MAX_WORKERS,
              epochs=config.EPOCHS,
              batch_size=config.BATCH_SIZE,
              t_equals_one=config.T_EQUALS_ONE):
        """
        TODO
        """
        def keep_iterating():
            result = True
            if max_train_time is not None:
                result &= (current_time - start_time) < max_train_time
            result &= current_error > max_train_error
            return result

        start_time = time.time()
        current_time = start_time
        current_error = float('inf') 

        while keep_iterating():
            self._buffer.extend(
                self._self_play(self_play_times, max_workers, t_equals_one)
            )

            size = min(len(self._buffer), batch_size)
            mini_batch = random.sample(self._buffer, size)
            
            x, y, z = zip(*mini_batch)
            x_train = np.array([
                self._convert_to_network_input(obs, to_play) for obs, to_play in list(x)]
            )
            train_prob = np.array(list(y))
            train_reward = np.array(list(z))

            history = self.neural_network.fit(x=x_train, y=[train_reward, train_prob],
                                              batch_size=32,
                                              epochs=epochs,
                                              verbose=2,
                                              validation_split=0)
            
            current_error = history.history['loss'][-1]
            current_time = time.time()

    def _self_play(self, games, max_workers, t_equals_one):
        def make_policy(env, nodes, counter):
            """Function used to generate the pi vector used to train AlphaZero's
                Neural Network.

                pi(a|s) = N(s,a)^(1/t) / Sum_b N(s,b)^(1/t)
                where t is a temperature parameter and b denotes all available
                actions at state s.

                If t = 1, it means a high level of exploration.
                If t -> 0, it means a low exploration.

            """
            
            visit_vector = np.zeros(env.action_space.n)
            for action, node in nodes.items():
                # TODO We are assuming action is an int or a tuple,
                #  check when generalizing
                visit_vector[action] = node.visit_count

            if counter > 0:
                # t = 1
                return visit_vector / visit_vector.sum() 
            else:
                # t -> 0
                index = np.argmax(visit_vector)
                pi = np.zeros(visit_vector.size)
                pi[index] = 1
                return pi

        def _self_play_(g, env, mcts):
            buffer = []

            for _ in range(g):
                observation = env.reset()
                game_states_data = []
                counter = t_equals_one

                s = time.time()

                while True:
                    action = mcts.move(observation)
                    counter = max(0, counter - 1)
                    pi = make_policy(env, mcts.stats['actions'], counter)
                    game_states_data.append(((observation, env.to_play), pi))
                    observation, _, done, _ = env.step(action)
                    mcts.update(action)

                    if done:
                        print(f"game finished in {time.time() - s}")
                        break

                perspective = 1
                # TODO turns may not switch every time, check when generalizing
                for i in range(len(game_states_data) - 1, -1, -1):
                    game_states_data[i] += (perspective * env.winner(),)
                    perspective *= -1

                buffer.extend(game_states_data)

            return buffer

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
        return self._mcts.move(observation)

    def update(self, action):
        self._mcts.update(action)

    def save(self, path):
        self.neural_network.save_model(path)

    def load(self, path):
        self.neural_network.load_model(path)

    def _value_function(self, node):
        nn_input = np.array([self._convert_to_network_input(node.observation, node.to_play)])
        predictions = self.neural_network.predict(nn_input)

        reward = predictions[0][0][0]
        probabilities = predictions[1][0]

        for i, child in node.children.items():
            child.probability = probabilities[i]

        return reward

    def _convert_to_network_input(self, board, to_play):
        black = board.copy()
        black[black == 1] = 0
        black[black == -1] = 1

        white = board.copy()
        white[white == -1] = 0

        player = 0 if to_play == 1 else 1 #All 1 if black to play, all 0 if white to play

        turn = np.full(board.shape, player)

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
        q_param = np.array([child.value for child in children])
        
        visits = np.array([child.visit_count for child in children])
        probabilities = np.array([child.probability for child in children])

        u_param = self.c_puct * probabilities * (
                np.sqrt(root.visit_count) / (1 + visits)
        )

        return np.argmax(q_param + u_param)


def create_alphazero(game, max_workers=None,
                     self_play_times=config.SELF_PLAY_TIMES,
                     max_train_time=config.MAX_TRAIN_TIME,
                     max_train_error=config.MAX_TRAIN_ERROR,
                     batch_size=config.BATCH_SIZE,
                     buffer_size=config.BUFFER_SIZE,
                     epochs=config.EPOCHS,
                     t_equals_one=config.T_EQUALS_ONE,
                     *args, **kwargs):

    if max_workers is None:
        raise NotImplementedError("not implemented yet")

    import ray

    if not ray.is_initialized():
        ray.init()

    def keep_iterating():
        result = True
        if max_train_time is not None:
            result &= (current_time - start_time) < max_train_time
        result &= current_error > max_train_error
        return result

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

    while keep_iterating():
        futures = [az._self_play.remote(g, None, t_equals_one)
                   for az, g in zip(azs, n_games)]
        moves = ray.get(futures)
        for m in moves:
            buffer.extend(m)

        size = min(len(buffer), batch_size)
        mini_batch = random.sample(buffer, size)

        x, y, z = zip(*mini_batch)
        x_train = np.array([
            actor._convert_to_network_input(obs, to_play) for obs, to_play in
            list(x)]
        )
        train_prob = np.array(list(y))
        train_reward = np.array(list(z))

        history = actor.neural_network.fit(x=x_train,
                                           y=[train_reward, train_prob],
                                           batch_size=32,
                                           epochs=epochs,
                                           verbose=2,
                                           validation_split=0)

        current_error = history.history['loss'][-1]
        current_time = time.time()

        weights = actor.get_weights()
        futures = [az.set_weights.remote(weights) for az in azs]
        ray.get(futures)

    for az in azs:
        ray.kill(az)

    return actor
