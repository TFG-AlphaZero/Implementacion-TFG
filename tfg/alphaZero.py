import sys
sys.path.insert(0, '/Documents/Juan Carlos/Estudios/Universidad/5º Carrera/TFG Informatica/ImplementacionTFG')

import numpy as np
import time
import tfg.alphaZeroConfig as config

from tfg.strategies import Strategy, argmax, MonteCarloTree
from tfg.util import play
from tfg.alphaZeroNN import NeuralNetworkAZ


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
                 epsilon=config.EPSILON,
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
        self.epsilon = epsilon

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.regularizer_constant = regularizer_constant
        self.momentum = momentum
        self.epochs = epochs
        self.residual_layers = residual_layers
        self.conv_filters = conv_filters
        self.conv_kernel_size = conv_kernel_size
        self.input_dim = self._env.observation_space.shape + config.INPUT_LAYERS
        self.output_dim = self._env.action_space.n

        self._selection_policy = QPlusU(self.c_puct)
        self._best_node_policy = BestNodePolicyAZ(
            self._env, self.t_equals_one, self.epsilon
        )

        self.mcts = MonteCarloTree(
            self._env,
            max_iter=self.mcts_times,
            selection_policy=self._selection_policy,
            value_function=self._value_function,
            best_node_policy=self._best_node_policy
        )

        self.buffer = []
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
            self.buffer.append(self._self_play(games=self.self_play_times))

            x, y, z = zip(*self.buffer)
            x_train = np.array([
                self._convert_to_network_input(obs) for obs in list(x)]
            )
            train_prob = np.array(list(y))
            train_reward = np.array(list(z))

            self.neural_network.fit(x=x_train, y=[train_reward, train_prob],
                                    batch_size=self.batch_size,
                                    epochs=self.epochs,
                                    verbose=2,
                                    validation_split=0)

            current_time = time.time()

    def _self_play(self, games=1, max_workers=None):
        def _self_play_(g):
            buffer = []

            for _ in range(g):
                observation = self.env.reset()
                self._best_node_policy.reset()
                game_states_data = []

                while True:
                    action = self.mcts.move(observation)
                    game_states_data.append((observation, self._best_node_policy.pi))
                    observation, _, done, _ = self.env.step(action)

                    if done:
                        break

                perspective = 1
                for i in range(len(game_states_data)-1, -1, -1) :
                    game_states_data[i] += (perspective * self.env.winner(),)
                    perspective *= -1
                buffer += game_states_data

            return buffer

        if max_workers is None:
            return _self_play_(games)

        d_games = games // max_workers
        r_games = games % max_workers
        n_games = [d_games] * max_workers
        if r_games != 0:
            for i in range(r_games):
                n_games[i] += 1

        results = Parallel(max_workers)(delayed(_self_play_)(g) for g in n_games)
        raise NotImplementedError
        #return reduce(lambda acc, x: map(sum, zip(acc, x)), results)

    def move(self, observation):
        return self.mcts.move(observation)

    def save(self, path):
        self.neural_network.save_model(path)

    def load(self, path):
        self.neural_network.load_model(path)

    def _value_function(self, node):
        nn_input = np.array([self._convert_to_network_input(node.observation)])
        predictions = self.neural_network.predict(nn_input)

        reward = predictions[0][0][0]
        probabilities = predictions[1][0]

        for i in node.children:
            node.children[i].probability = probabilities[i]

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


class BestNodePolicyAZ:
    """Class representing the Best Node Policy used by AlphaZero.
    
    It is used at the end of MCTS algorithm to select the returned action.
    Selects the child which maximises:
    pi(a|s) = N(s,a)^(1/t) / Sum_b N(s,b)^(1/t)
    where t -> 0 is a temperature parameter and b denotes the available action
    at state s.

    If t = 1, this means a high-level of exploration.
    If t -> 0, this means a low exploration.

    When playing a real game with an opponent, the temperature is set to t -> 0
    all the time.
    
    It is a functional class: ([MonteCarloTreeNode]) -> [int] which returns
    pi(a|s) for each action
    """

    def __init__(self, env, t_equals_one, epsilon):
        """
        TODO
        Args:
            env ():
            t_equals_one (int): Times that t = 1. When called for the
            (t_equal_one) th time, it will become t -> 0.
            epsilon ():

        """

        self._env = env
        self.t_equals_one = t_equals_one
        self.counter = t_equals_one
        self.epsilon = epsilon
        self.pi = None

    def __call__(self, nodes):
        # TODO Recordar cambiar esto! Es para que no pete. Ademas, tenemos el
        #       problema de como hacer reset cuando jugamos varios juegos.
        t = 1
        # FIXME Peta overflow si hago el t mas pequeno o visit_count se hace > 3
        # t = 1 if self.counter > 0 else self.epsilon
        self.counter = max(0, self.counter - 1)

        """visit_vector = []
        for action in range(self._env.action_space.n): #Hay que cambiarlo
            if action in nodes:
                visit_vector.append(nodes[action].visit_count)
            else:
                visit_vector.append(0)"""

        visit_vector = np.zeros(self._env.action_space.shape)
        # TODO esto probablemente mejor fuera que aquí no pega mucho
        for action, node in nodes.items():
            # We are assuming action is an int or a tuple
            visit_vector[action] = node.visit_count

        # fun1 = lambda i: i**(1/t)
        # fun2 = np.vectorize(fun1)

        sum = np.sum(visit_vector ** (1 / t))
        self.pi = visit_vector / sum
        # self.pi = [fun1(visits) / sum for visits in visit_vector]

        return np.argmax(self.pi)
    
    def reset(self):
        self.counter = self.t_equals_one
