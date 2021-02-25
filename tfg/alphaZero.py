import sys
sys.path.insert(0, '/Documents/Juan Carlos/Estudios/Universidad/5º Carrera/TFG Informatica/ImplementacionTFG')

import numpy as np

from tfg.strategies import Strategy, argmax, MonteCarloTree
from tfg.util import play

class AlphaZero(Strategy):
    """Game strategy implementing AlphaZero algorithm.

    Parameters setting:
    c_puct
    MCTS times
    self_play_times
    Residual blocks
    Batch size
    Learning rate
    Optimizer
    Dirichlet noise
    Weight of noise
    t = 1 for the first n moves

    """

    def __init__(self, env, max_training_time = None, 
                 c_puct = None, 
                 mcts_times = None,
                 self_play_times = None,
                 residual_blocks = None,
                 batch_size = None,
                 learning_rate = None,
                 t_equals_one = None):
        """

        Args:
            ...

        """
        
        if c_puct is None:
            c_puct = 5

        if mcts_times is None:
            mcts_times = 800

        if self_play_times is None:
            self_play_times = 25_000
        
        if t_equals_one is None:
            t_equals_one = 30

        self._env = env
        self.mcts_times = mcts_times
        self.self_play_times = self_play_times
        self.t_equals_one = t_equals_one

        self._selection_policy = QPlusU(c_puct)
        self._best_node_policy = BestNodePolicyAZ(self.t_equals_one)

        self.buffer = []
        self.neural_network = [] #Implementar
        self.mcts = MonteCarloTree(self._env, max_iter=self.mcts_times, max_time=None,
                                    selection_policy=None, value_function=None,
                                    simulation_policy=None, update_function=None, best_node_policy=self._best_node_policy)


    @property
    def env(self):
        """tfg.games.GameEnv: Game this strategy is for."""
        return self._env

    def train(self):
        """

        """

        #while True : #Cambiar por timeLeft o error > threshold
        
        self.buffer += self._self_play(games=self.self_play_times)

            #Neural Network
            #entrenar network

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
                
                buffer.append(GameData(game_states_data, self.env.winner))

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

class GameData:
    """
    Class representing the data to store from a single game in order to train the neural network

    """

    def __init__(self, states = None, result = None):
        if states is None:
            states = []
        
        self._states = states
        self._result = result

    def __str__(self):
        for state in states:
            print(state)
        print(result)
        

class QPlusU:
    """
    Class representing the Q + U formula. (Deep RL)
    
    In the step of select, the action is selected by the formula 
    a = argmax(Q(s,a) + U(s,a)) where Q(s,a) = W/N encourages 
    the exploitation and U(s,a) = c_puct * P(s,a) * sqrt(Sum_b N(s,b)) / 1 + N(s,a)
    encourages the exploration. 
    
    c_puct is a parameter determining the exploration scale (5 for AlphaZero).
    """

    def __init__(self, c_puct):
        """

        Args:
            c_puct (float): Exploration scale constant.

        """
        self.c_puct = c_puct

    def __call__(self):
        
        raise NotImplementedError

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

    def __init__(self, t_equals_one):
        """

        Args:
            t_equals_one (int): Times that t = 1. When called for the (t_equal_one) th time,
            it will become t -> 0

        """

        self.t_equals_one = t_equals_one
        self.counter = t_equals_one

    def __call__(self, nodes):
        t = 1 if self.counter > 0 else 0.01 #Peta overflow si hago el t mas pequeno.
        self.counter = max(0, self.counter - 1)

        fun1 = lambda i : i.visit_count**(1/t)
        fun2 = np.vectorize(fun1)

        sum = np.sum(fun2(nodes))
        pi = [fun1(node) / sum for node in nodes]

        return np.argmax(pi)
    
    def reset(self):
        self.counter = self.t_equals_one
