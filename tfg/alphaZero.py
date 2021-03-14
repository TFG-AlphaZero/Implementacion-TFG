import sys
sys.path.insert(0, '/Documents/Juan Carlos/Estudios/Universidad/5º Carrera/TFG Informatica/ImplementacionTFG')

import numpy as np
import collections
import time
import itertools
import random
import tfg.alphaZeroConfig as config

from tfg.strategies import Strategy, argmax, MonteCarloTree
from tfg.util import play
from tfg.alphaZeroNN import NeuralNetworkAZ
from joblib import Parallel, delayed

class AlphaZero(Strategy):
    """Game strategy implementing AlphaZero algorithm."""

    # TODO podemos crear una clase Config o algo así y se le pasa directamente
    #       en vez de tantos parámetros aquí
    #       Para los que sean que se pasen a MCTS o así también se puede usar
    #       **kwags (por ejemplo si se le quiere dar un max_time o algo así)
    def __init__(self,
                 env,
                 c_puct=config.C_PUCT,
                 mcts_times=config.MCTS_TIMES,
                 self_play_times=config.SELF_PLAY_TIMES,
                 t_equals_one=config.T_EQUALS_ONE,
                 buffer_size = config.BUFFER_SIZE,
                 max_workers = config.MAX_WORKERS,
                 batch_size=config.BATCH_SIZE,
                 learning_rate=config.LEARNING_RATE,
                 regularizer_constant=config.REGULARIZER_CONST,
                 momentum=config.MOMENTUM,
                 epochs=config.EPOCHS,
                 residual_layers=config.RESIDUAL_LAYERS,
                 conv_filters=config.CONV_FILTERS,
                 conv_kernel_size=config.CONV_KERNEL_SIZE):
        """

        Args:
            Me da pereza escribirlos todos xd

        """

        self._env = env
        # TODO Quizas sea innecesario guardar todos estos parametros y
        #       enchufarlos directamente al mcts y neural network sin guardarlos
        #       en atributos.
        self.c_puct = c_puct
        self.mcts_times = mcts_times
        self.self_play_times = self_play_times
        self.t_equals_one = t_equals_one
        self.counter = t_equals_one
        self.buffer_size = buffer_size
        self.max_workers = max_workers

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.regularizer_constant = regularizer_constant
        self.momentum = momentum
        self.epochs = epochs
        self.residual_layers = residual_layers
        self.conv_filters = conv_filters
        self.conv_kernel_size = conv_kernel_size
        self.input_dim = self._env.observation_space.shape + config.INPUT_LAYERS
        # FIXME ahora no funciona con esto
        #   Falla al crear el vector de probabilidades, pero igualmente
        #   habría que especificar esto de otra forma.
        #   Por ejemplo en el tres en raya igual nos conviene que la salida
        #   sea de 3x3x1 para que coincida con juegos como el ajedrez (8x8x63)
        #   y no que cada juego tenga una salida distinta.
        #   Con esto lo que pasa es que quizá habría que transformar las
        #   acciones de alguna forma, como por ejemplo pasar del 3x3x1 a 9 en
        #   _value_function.
        self.output_dim = self._env.action_space.n

        self._selection_policy = QPlusU(self.c_puct)

        self.mcts = MonteCarloTree(
            self._env,
            max_iter=self.mcts_times,
            selection_policy=self._selection_policy,
            value_function=self._value_function,
            best_node_policy='robust'
        )

        self.buffer = collections.deque(maxlen=self.buffer_size)
        
        self.neural_network = NeuralNetworkAZ(
            learning_rate=self.learning_rate,
            regularizer_constant=self.regularizer_constant,
            momentum=self.momentum,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            residual_layers=self.residual_layers,
            filters=self.conv_filters,
            kernel_size=self.conv_kernel_size
        )

    @property
    def env(self):
        """tfg.games.GameEnv: Game this strategy is for."""
        return self._env

    def train(self, max_train_time=config.MAX_TRAIN_TIME,
              max_train_error=config.MAX_TRAIN_ERROR):
        """

        """
        def keep_iterating():
            result = True
            if max_train_time is not None:
                result &= (current_time - start_time) < max_train_time
            result &= error > max_train_error
            return result

        start_time = time.time()
        current_time = start_time
        error = 1

        while keep_iterating():
            self.buffer.extend(self._self_play(games=self.self_play_times, max_workers=self.max_workers))

            mini_batch = random.sample(self.buffer, min(len(self.buffer), self.batch_size))
            
            x, y, z = zip(*mini_batch)
            x_train = np.array([
                self._convert_to_network_input(obs) for obs in list(x)]
            )
            train_prob = np.array(list(y))
            train_reward = np.array(list(z))

            self.neural_network.fit(x=x_train, y=[train_reward, train_prob],
                                    batch_size=32,
                                    epochs=self.epochs,
                                    verbose=2,
                                    validation_split=0)

            current_time = time.time()

    def _self_play(self, games=1, max_workers=None):
        def make_policy(nodes):
            """Function used to generate the pi vector used to train AlphaZero's Neural Network.

            pi(a|s) = N(s,a)^(1/t) / Sum_b N(s,b)^(1/t)
            where t is a temperature parameter and b denotes all available actions at state s.

            If t = 1, it means a high-level of exploration.
            If t -> 0, it means a low exploration.
            """

            self.counter = max(0, self.counter - 1)
            
            visit_vector = np.zeros(self._env.action_space.n)
            for action, node in nodes.items():
                # TODO We are assuming action is an int or a tuple, check when generalizing
                visit_vector[action] = node.visit_count

            if self.counter > 0 : # t = 1
                return visit_vector / visit_vector.sum() 
            else : # t -> 0
                index = np.argmax(visit_vector)
                pi = np.zeros(visit_vector.size)
                pi[index] = 1
                return pi

        def _self_play_(g):
            buffer = []

            for _ in range(g):
                observation = self._env.reset()
                game_states_data = []
                self.counter = self.t_equals_one

                while True:
                    action = self.mcts.move(observation)
                    pi = make_policy(self.mcts.stats['actions'])
                    game_states_data.append((observation, pi))
                    observation, _, done, _ = self._env.step(action)
                    self.mcts.update(action)

                    if done:
                        break

                perspective = 1
                # TODO turns may not switch every time, check when generalizing
                for i in range(len(game_states_data) - 1, -1, -1):
                    game_states_data[i] += (perspective * self._env.winner(),)
                    perspective *= -1

                buffer.extend(game_states_data)

            return buffer

        if max_workers is None:
            return _self_play_(games)

        d_games = games // max_workers
        r_games = games % max_workers
        n_games = [d_games] * max_workers
        if r_games != 0:
            for i in range(r_games):
                n_games[i] += 1

        bufs = Parallel(max_workers, backend = 'threading')(delayed(_self_play_)(g) for g in n_games)
        return list(itertools.chain.from_iterable(bufs))

    def move(self, observation):
        return self.mcts.move(observation)

    def update(self, action):
        return self.mcts.update(action)

    def save(self, path):
        self.neural_network.save_model(path)

    def load(self, path):
        self.neural_network.load_model(path)

    def _value_function(self, node):
        nn_input = np.array([self._convert_to_network_input(node.observation)])
        predictions = self.neural_network.predict(nn_input)

        reward = predictions[0][0][0]
        probabilities = predictions[1][0]

        for i, child in node.children.items():
            child.probability = probabilities[i]

        return reward

    def _convert_to_network_input(self, observation):
        def convert_to_binary(board):
            first = board.copy()
            first[first == -1] = 0

            second = board.copy()
            second[second == 1] = 0
            second[second == -1] = 1

            return np.append(first, second)

        input = convert_to_binary(observation)
        input = np.reshape(input, self.input_dim)

        return input


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