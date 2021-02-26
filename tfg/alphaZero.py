import sys
sys.path.insert(0, '/Documents/Juan Carlos/Estudios/Universidad/5º Carrera/TFG Informatica/ImplementacionTFG')

import numpy as np
import time
import tfg.alphaZeroConfig as config

from tfg.strategies import Strategy, argmax, MonteCarloTree
from tfg.util import play
from tfg.alphaZeroNN import NeuralNetwork


class AlphaZero(Strategy):
    """Game strategy implementing AlphaZero algorithm.

    """

    def __init__(self, env,
                 c_puct = config.C_PUCT, mcts_times = config.MCTS_TIMES, self_play_times = config.SELF_PLAY_TIMES,  t_equals_one = config.T_EQUALS_ONE, epsilon = config.EPSILON,
                 batch_size = config.BATCH_SIZE, learning_rate = config.LEARNING_RATE, regularizer_constant = config.REGULARIZER_CONST, momentum = config.MOMENTUM, 
                 epochs = config.EPOCHS, residual_layers = config.RESIDUAL_LAYERS, conv_filters = config.CONV_FILTERS, conv_kernel_size = config.CONV_KERNEL_SIZE):
        """

        Args:
            Me da pereza escribirlos todos xd

        """
        

        self._env = env
        #Quizas sea innecesario guardar todos estos parametros y enchufarlos directamente al mcts y neural network sin guardarlos en atributos.
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

        self._selection_policy = QPlusU(self.c_puct)
        self._best_node_policy = BestNodePolicyAZ(self.t_equals_one, self.epsilon)

        self.mcts = MonteCarloTree(self._env, max_iter=self.mcts_times, max_time=None,
                                    selection_policy=self._selection_policy, value_function=self._value_function,
                                    simulation_policy=None, update_function=None, best_node_policy=self._best_node_policy)

        self.buffer = []
        self.neural_network = NeuralNetwork(learning_rate=self.learning_rate, regularizer_constant = self.regularizer_constant, momentum = self.momentum,
                                            input_dim = config.INPUT_LAYERS + self._env.observation_space.shape, output_dim = self._env.action_space.n, 
                                            residual_layers=self.residual_layers, filters = self.conv_filters, kernel_size=self.conv_kernel_size)

    @property
    def env(self):
        """tfg.games.GameEnv: Game this strategy is for."""
        return self._env

    def train(self, max_train_time = None, max_train_error = 0.01):
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

        #while keep_iterating() :
            
        self.buffer += self._self_play(games=self.self_play_times)

        #Neural Network
        
        current_time = time.time()

        return self.buffer

    def _self_play(self, games = 1, max_workers = None):
        def _self_play_(g):
            buffer = []

            for _ in range(g):
                observation = self.env.reset()
                game_states_data = []

                while True:
                    action = self.mcts.move(observation)
                    game_states_data.append((observation, 10)) #Guardar las prob del nodo
                    observation, _, done, _ = self.env.step(action)

                    if done:
                        break
                
                buffer.append(GameData(game_states_data, self.env.winner()))

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
        """

        """

        raise NotImplementedError

    def _value_function(self, node):
        prob, rew = self.neural_network.predict(node.observation)
        
        for child in node.children:
            child.probability = prob

        return 1



class GameData:
    """
    Class representing the data to store from a single game in order to train the neural network

    """

    def __init__(self, states = None, result = None):
        if states is None:
            states = []
        
        self._states = states
        self._result = result

    def __repr__(self):
        return '(' + str(self._states) + ', ' + str(self._result) + ')'
        

class QPlusU:
    """
    Class representing the Q + U formula. (Deep RL)
    
    In the step of select, the action is selected by the formula 
    a = argmax(Q(s,a) + U(s,a)) where Q(s,a) = W/N encourages 
    the exploitation and U(s,a) = c_puct * P(s,a) * sqrt(Sum_b N(s,b)) / 1 + N(s,a)
    encourages the exploration. 
    
    c_puct is a parameter determining the exploration scale (5 for AlphaZero).

    (root: MonteCarloTreeNode, children: [MonteCarloTreeNode]) -> selected_index: int,
    """



    def __init__(self, c_puct):
        """
        Args:
            c_puct (float): Exploration scale constant.

        """
        self.c_puct = c_puct

    def __call__(self, root, children):
        q_param = np.array([child.value for child in children])
        
        children_visit_count = np.array([child.visit_count for child in children])
        children_probabilities = np.array([child.probability for child in children])
        u_param = c_puct * children_probabilities * np.sqrt(root.visit_count) / (1 + children_visit_count)

        return np.argmax(q_param + u_param) 


"""
    def __call__(self, root, children):
        values = np.array([child.value for child in children])
        visits = np.array([child.visit_count for child in children])
        if (visits == 0).any():
            # Avoid divide by zero and choose one with visits == 0
            return np.random.choice(np.where(visits == 0)[0])
        uct_values = values + self.c * np.sqrt(
            np.log(root.visit_count) / visits
        )
        return np.argmax(uct_values)

"""

class BestNodePolicyAZ:
    """
    Class representing the Best Node Policy used by AlphaZero.
    
    It is used at the end of MCTS algorithm to select the returned action.
    Selects the child which maximises:
    pi(a|s) = N(s,a)^(1/t) / Sum_b N(s,b)^(1/t)
    where t -> 0 is a temperature parameter and b denotes the available action at state s.

    If t = 1, this means a high-level of exploration.
    If t -> 0, this means a low exploration.

    When playing a real game with an opponent, the temperature is set to t -> 0 all the time.
    
    It is a functional class: ([MonteCarloTreeNode]) -> [int]
    which returns pi(a|s) for each action
    """

    def __init__(self, t_equals_one, epsilon):
        """

        Args:
            t_equals_one (int): Times that t = 1. When called for the (t_equal_one) th time,
            it will become t -> 0

        """

        self.t_equals_one = t_equals_one
        self.counter = t_equals_one
        self.epsilon = epsilon

    def __call__(self, nodes):
        t = 1 if self.counter > 0 else self.epsilon #Peta overflow si hago el t mas pequeno.
        self.counter = max(0, self.counter - 1)

        fun1 = lambda i : i.visit_count**(1/t)
        fun2 = np.vectorize(fun1)

        sum = np.sum(fun2(nodes))
        pi = [fun1(node) / sum for node in nodes]

        return np.argmax(pi)
    
    def reset(self):
        self.counter = self.t_equals_one
