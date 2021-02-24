import sys
sys.path.insert(0, '/Documents/Juan Carlos/Estudios/Universidad/5º Carrera/TFG Informatica/ImplementacionTFG')

from tfg.strategies import Strategy, argmax, MonteCarloTree

class AlphaZero(Strategy):
    """Game strategy implementing AlphaZero algorithm.

    Parameters setting:
    c_puct
    MCTS times
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
        
        if t_equals_one is None:
            t_equals_one = 30

        self._env = env
        self.mcts_times = mcts_times
        self.t_equals_one = t_equals_one

        self._selection_policy = QPlusU(c_puct)

        self.neural_network = []
        self.mcts = MonteCarloTree(env, mcts_times, None, self._selection_policy, )


    @property
    def env(self):
        """tfg.games.GameEnv: Game this strategy is for."""
        return self.env

    def train(self):
        """

        """

        raise NotImplementedError



    def move(self, observation):
        """

        """

        raise NotImplementedError

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

class ProbVisitCount:
    """
    Class representing the best node policy used by AlphaZero.
    
    It is used at the end of MCTS algorithm to select the returned action.
    Selects the child which maximises:
    pi(a|s) = N(s,a)^(1/t) / Sum_b N(s,b)^(1/t)
    where t -> 0 is a temperature parameter and b denotes the available action at state s.

    If t = 1, this means a high-level of exploration.
    If t -> 0, this means a low exploration.

    When playing a real game with an opponent, the temperature is set to t -> 0 all the time.
    
    It is a functional class: ([MonteCarloTreeNode]) -> ([int], int)
    Stores pi(a|s) and the index which maximixes it
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
        t = 1 if self.counter > 0 else 0.001 #Mirar si es mejor usar una constante a pelo o epsilon!
        self.counter = max(0, self.counter - 1)

        raise NotImplementedError
    
    def reset(self):
        self.counter = self.t_equals_one

