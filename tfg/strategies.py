import sys
sys.path.insert(0, '/Documents/Juan Carlos/Estudios/Universidad/5º Carrera/TFG Informatica/ImplementacionTFG')

import abc
import copy
import random
import re
import time

import gym
import numpy as np
import scipy.special as sp

import tfg.games


def argmax(values, key=None):
    key_ = (lambda i: values[i]) if key is None else (lambda i: key(values[i]))
    return max(range(len(values)), key=key_)


def argmin(values, key=None):
    key_ = (lambda i: values[i]) if key is None else (lambda i: key(values[i]))
    return min(range(len(values)), key=key_)


class Strategy(abc.ABC):
    """Interface for game policies.

    It implements the move method which maps observations to an action.
    """

    def move(self, observation):
        """Returns the action that the agent should taken given the current
        observation. It acts like the policy function.

        Args:
            observation (object): Current state of the game.

        Returns:
            object: Action that the agent should take.

        """
        raise NotImplementedError

    def update(self, action):
        """Reports that the given action has been taken so that this Strategy
        can update itself. Optional method, subclasses may implement it or not.

        Args:
            action (object): Action taken. None means that this Strategy must
            reset everything.

        """
        pass


class HumanStrategy(Strategy):
    """Implementation of Strategy so that humans can play a game."""

    def __init__(self, env, name='PLAYER'):
        """

        Args:
            env (tfg.games.GameEnv): Game this strategy is for.
            name (object, optional): Name that will be printed for this player.
                Defaults to 'PLAYER'.

        """
        self.env = env
        self.name = str(name)

    def move(self, observation=None):
        """Renders the game and asks the player to input a move.

        Args:
            observation (object): Ignored as the player will see the current
                state.

        Returns:
            object: Action chosen by the human player.

        Raises:
            ValueError: If the action is not a valid action in the action space
                of the environment (the game).

        """
        self.env.render(mode='human')
        legal_actions = self.env.legal_actions()
        t = self.env.action_space.dtype.type
        print()
        print("[" + self.name + "] ", end="")
        print(f"choose action: {', '.join(map(str, legal_actions))}")
        action = re.split(r'\s+', input("> "))
        # Try to cast the action to t or array[t]
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            action = action[0]
            action = t(action)
        else:
            action = np.array(t(a) for a in action)
        if action not in self.env.action_space:
            raise ValueError(f"given action {action}"
                             f"is not contained in the action space")
        return action


class Minimax(Strategy):
    """Game strategy implementing Minimax algorithm and AlphaBeta prune.

    Minimax explores the whole game tree and tries to maximize each player's
    reward. That is, for WHITE, also called the max player, the algorithm
    will try to maximize its payoff while trying to minimize it for BLACK,
    the min player. As a remainder, positive reward means WHITE is winning,
    while negative means BLACK is winning.

    AlphaBeta prune is an improvement of Minimax algorithm, where parts of
    the tree where we know we can't get a better value get pruned directly.

    """

    def __init__(self, env, max_depth=None, heuristic=None, alpha_beta=True,
                 ordering=lambda x: random.random()):
        """

        Args:
            env (tfg.games.GameEnv): Game this strategy is for.
            max_depth (int, optional): Maximum depth the tree is allowed to
            grow. If None it will grow until a
                leaf node is reached. If set, heuristic must be given. Defaults
                to None.
            heuristic (function, optional): If max_depth is not None, heuristic
                function (observation: object, to_play: int) -> int that will be
                called to estimate the value of a leaf node. This value will
                be positive if WHITE is more likely to win, negative if BLACK is
                the one who is winning and 0 if the game is estimated to end
                in a draw. It is recommended that the return value of this
                function is between -1 and 1.
            alpha_beta (bool, optional): Determines whether or not to use
                AlphaBeta prune. Defaults to True.
            ordering (function, optional): Function (action: object) -> number
                that will be used as key function to sort actions before
                traversing them. Lower numbers go first. If set to None,
                actions will be traversed with the same order as they were
                returned by env.legal_actions() function. Defaults to
                random.

        """

        if max_depth is not None and heuristic is None:
            raise ValueError("max_depth was given but heuristic wasn't")

        self._env = env
        self.max_depth = max_depth
        self._heuristic = heuristic
        self.alpha_beta = alpha_beta
        self.ordering = ordering

    @property
    def env(self):
        """tfg.games.GameEnv: Game this strategy is for."""
        return self._env

    def move(self, observation):
        """Implementation of the Minimax algorithm.

        Args:
            observation (object): Current state of the game (must be the same as
                the one in self.env or the algorithm.
            might not work).

        Returns:
            object: Action that the agent should take.

        """
        player = self._env.to_play
        action, _ = (self._minimax(self._env, observation, player)
                     if not self.alpha_beta
                     else self._alpha_beta(self._env, observation, player))
        return action

    def _minimax(self, env, observation, player, depth=0):
        to_play = env.to_play
        if self.max_depth is not None and depth == self.max_depth:
            # Use the heuristic if we reached the maximum depth
            return None, self._heuristic(observation, to_play)

        legal_actions = self._legal_actions(env)
        selected_action = None
        value = None

        for action in legal_actions:
            env_ = copy.deepcopy(env)
            obs, reward, done, _ = env_.step(action)

            if not done:
                _, reward = self._minimax(env_, obs, player, depth + 1)

            if (selected_action is None or
                    self._better_value(value, reward, player, to_play)):
                selected_action = action
                value = reward

        # Return the selected action at the root
        return selected_action, value

    def _alpha_beta(self, env, observation, player,
                    depth=0, alpha=None, beta=None):
        to_play = env.to_play
        if self.max_depth is not None and depth == self.max_depth:
            # Use the heuristic if we reached the maximum depth
            return None, self._heuristic(observation, to_play)

        legal_actions = self._legal_actions(env)
        selected_action = None

        if player == tfg.games.WHITE and player == to_play or \
                player == tfg.games.BLACK and player != to_play:
            # Alpha
            for action in legal_actions:
                env_ = copy.deepcopy(env)
                obs, reward, done, _ = env_.step(action)

                if not done:
                    _, reward = self._alpha_beta(env_, obs, player,
                                                 depth + 1, alpha, beta)

                if alpha is None or alpha < reward:
                    selected_action = action
                    alpha = reward
                    if beta is not None and alpha >= beta:
                        break
            return selected_action, alpha
        else:
            # Beta
            for action in legal_actions:
                env_ = copy.deepcopy(env)
                obs, reward, done, _ = env_.step(action)

                if not done:
                    _, reward = self._alpha_beta(env_, obs, player,
                                                 depth + 1, alpha, beta)

                if beta is None or beta > reward:
                    selected_action = action
                    beta = reward
                    if alpha is not None and alpha >= beta:
                        break
            return selected_action, beta

    @staticmethod
    def _better_value(current, new, player, to_play):
        # Work with (win, >0), (lose, <0)
        current = player * current
        new = player * new
        if player == to_play and new > current:
            return True
        if player != to_play and new < current:
            return True
        return False

    def _legal_actions(self, env):
        legal_actions = env.legal_actions()
        # Sort if necessary
        if self.ordering is not None:
            legal_actions.sort(key=self.ordering)
        return legal_actions


class MonteCarloTreeNode(dict):
    """Node used during the Monte Carlo Tree Search algorithm."""

    def __init__(self, observation, to_play):
        """

        Args:
            observation (object): State of the game this node is representing.
            to_play (int): Player to play in this state, either BLACK or WHITE.

        """
        super(MonteCarloTreeNode, self).__init__()
        self.observation = observation
        self.to_play = to_play
        self.visit_count = 0
        self.value = 0
        self.value_sum = 0
        self.value_squared_sum = 0
        self.value_variance = np.inf
        self.children = dict()

    def expanded(self):
        """Returns whether this node has already been expanded or not.

        Returns:
            bool: True if this node has already been expanded and False
                otherwise.

        """
        return bool(self.children)

    def expand(self, env, update_fun=None):
        """Expands this node. This means that the dict of children gets
        initialized.

        Args:
            env (tfg.games.GameEnv): Game containing this node's state.
            update_fun (function, optional): Void function that takes newly
                created MonteCarloTreeNode and initializes custom statistics.

        """
        actions = env.legal_actions()
        for action in actions:
            env_ = copy.deepcopy(env)
            observation, _, _, info = env_.step(action)
            to_play = env_.to_play
            node = MonteCarloTreeNode(observation, to_play)
            if update_fun is not None:
                update_fun(node)
            self.children[action] = node

    def update(self, value, update_fun=None):
        """Updates the attributes of this node.

        Args:
            value (float): Observed value that will update this node's info.
            update_fun (function, optional): Void function that takes this
                MonteCarloTreeNode and updates custom statistics.

        """
        self.visit_count += 1
        self.value_sum += value
        self.value = self.value_sum / self.visit_count
        self.value_squared_sum += (value - self.value) ** 2
        if self.visit_count > 1:
            self.value_variance = (
                    self.value_squared_sum / (self.visit_count - 1)
            )
        if update_fun is not None:
            update_fun(self)

    def __eq__(self, other):
        return (self.observation == other.observation and
                self.to_play == other.to_play)

    def __hash__(self):
        return hash((self.observation, self.to_play))


class MonteCarloTree(Strategy):
    """Game strategy (policy) implementing Monte Carlo Tree Search algorithm

    This algorithm was first described in Coulom, 2006; Kocsis and Szepesvári,
    2006; and Chaslot et al., 2006a.

    Essentially, for every move of the player the algorithm will run max_iter
    iterations (during max_time seconds at most) of the following:
    1. Selection:       starting in the root traverse down the tree using a
                        given selection_policy until it encounters a
                        non-expanded node.
    2. Expansion:       add all children of the non-expanded node to the tree.
    3. Simulation:      self-play the rest of the game randomly (only if
                        value_function has not been set).
    4. Backpropagation: knowing the result of the game (the reward), update the
                        value and other stats of every node in the path between
                        the root and the expanded node using the
                        update_function.

    After max_iter iterations or max_time seconds it selects an action that
    results in the node that maximizes the value returned by best_node_policy.

    """

    def __init__(self, env, max_iter=None, max_time=None,
                 selection_policy=None,
                 value_function=None,
                 update_function=None,
                 best_node_policy=None,
                 reset_tree=True):
        """

        Args:
            env (tfg.games.GameEnv): Game this strategy is for.
            max_iter (int, optional): Total number of iterations of the
                algorithm for each move. If not set, the algorithm will run
                until there is no time left. Either max_iter or max_time (or
                both) must be set.
            max_time (float, optional): Maximum amount of seconds the
                algorithm will be running for each move. If not set, the
                algorithm will run exactly max_iter iterations. Time spent
                during selection of the final move will not be taken into
                account. Either max_iter or max_time (or both) must be set.
            selection_policy (function or str, optional): Function that will
                be used to select a child during selection phase Can be
                either a function
                    (root: MonteCarloTreeNode,
                    children: [MonteCarloTreeNode]) -> selected_index: int,
                or a string representing the function to apply, one of the
                following: {'random', 'uct', 'omc', 'pbbm'}, where 'uct' is
                tfg.strategies.UCT with C=sqrt(2), 'omc' is
                tfg.strategies.OMC and 'pbbm' is tfg.strategies.PBBM.
                Defaults to 'uct'. If a functional class is given it can also
                define the value_range (int, int) attribute which will tell the
                backpropagation algorithm which value to pass, linearly
                transforming original rewards returned by the game or
                value_function from [-1, 1] to value_range.
            value_function (function, optional): Function that will be used
                to compute the estimated value of a newly expanded node. Must
                be a function
                    (node: MonteCarloTreeNode) -> value: number
                that may also modify any of the custom attributes of the node.
                If this parameter is None the value will be estimated by
                simulating a game starting from the expanded node. Otherwise,
                the given function will be used and simulation_policy will be
                ignored. If a functional class is given it can also define
                the value_range (int, int) attribute specifying the minimum
                (loss) and maximum (win) value the function can return.
            update_function (function, optional): Function that will be used
                to update the value of custom statistics of each node during
                node creation and backpropagation. Must be a void function that
                takes MonteCarloTreeNode as its unique argument. It is up to
                the caller to make sure that custom attributes are properly
                created during node's construction.
            best_node_policy (function or str, optional): Function that will
                be used to select the returned action at the end of the
                algorithm. Can be either a function
                    (nodes: [MonteCarloTreeNode]) -> index: number
                or a string representing the function to apply, one of the
                following: {'robust', 'max', 'secure'}, where 'robust'
                selects the node with higher visit count, 'max' the one with
                highest value and 'secure' uses tfg.strategies.SecureChild
                formula with A=4. Defaults to 'robust'.
            reset_tree (bool, optional): Whether to reset the game tree after
                each call to move or keep it for the next call. Defaults to
                True (reset). If set to False all moves must be reported via
                the update method.

        """
        if max_iter is None and max_time is None:
            raise ValueError("at least one of max_iter and max_time "
                             "must be not None")

        if selection_policy is None:
            selection_policy = 'uct'

        if best_node_policy is None:
            best_node_policy = 'robust'

        self._env = env
        self.max_iter = max_iter
        self.max_time = max_time

        self._selection_policy = (
            self._selection_policy_from_string(selection_policy)
            if isinstance(selection_policy, str)
            else selection_policy
        )

        self._value_function = value_function

        self._update_function = update_function

        self._best_node_policy = (
            self._best_node_policy_from_string(best_node_policy)
            if isinstance(best_node_policy, str)
            else best_node_policy
        )

        self._expected_value_range = (
            self._selection_policy.value_range
            if hasattr(self._selection_policy, 'value_range')
            else None
        )

        self._observed_value_range = (
            self._value_function.value_range
            if self._value_function is not None and
            hasattr(self._value_function, 'value_range')
            else [-1, 1]
        )

        self.reset_tree = reset_tree
        self._root = None

        self._stats = dict()

    @property
    def env(self):
        """tfg.games.GameEnv: Game this strategy is for."""
        return self._env

    @property
    def stats(self):
        """dict: Dict containing stats of the last execution of move:

            'actions' -> dict from action to MonteCarloTreeNode (containing
                stats, such as observation, value or visit_count).
            'iters' -> number of iterations.
            'time' -> time spent int seconds.

        """
        return self._stats

    def move(self, observation):
        """Implementation of the MCTS algorithm.

        Args:
            observation (object): Current state of the game (must be the same
                as the one in self.env or the algorithm might not work).

        Returns:
            object: Action that the agent should take.

        """

        def time_left():
            result = True
            if self.max_iter is not None:
                result &= i < self.max_iter
            if self.max_time is not None:
                result &= (current_time - start) + iter_time < self.max_time
            return result

        start = time.time()
        current_time = start
        iter_time = 0

        player = self._env.to_play

        # Initialize tree
        # TODO exploration noise?
        if self._root is None:
            root = MonteCarloTreeNode(observation, player)
            if not self.reset_tree:
                self._root = root
        else:
            root = self._root

        i = 0
        # Iterate the algorithm max_iter times
        while time_left():
            # Reset search every iteration
            history = [root]
            env = copy.deepcopy(self._env)
            done = False
            reward = 0

            current_node = root

            # Selection phase
            while not done and current_node.expanded():
                # Select next child
                current_node, action = self._select(env, current_node)

                # Take the action
                _, reward, done, _ = env.step(action)
                # Update reward: if WHITE won reward=1,
                # but if we are black reward should be -1 as we lost
                reward *= player
                # TODO Fix reward perspective
                # reward *= current_node.to_play
                
                history.append(current_node)

            # We might have found a real leaf node during selection
            # so there is no need to expand or simulate
            if not done:
                # Expansion phase
                current_node.expand(env, self._update_function)

                if self._value_function is None:
                    # Simulation phase
                    reward = self._simulate(env) * player
                else:
                    # Estimate via value_function
                    #TODO Fix reward perspective
                    reward = self._value_function(current_node)

            # Backpropagation phase
            # Who played the move that lead to that node
            """
            perspective = -1
            for node in reversed(history):
                r = reward * perspective
                self._backpropagate(node, r)
                perspective *= -1
            """
            #TODO Fix reward perspective
            to_play = player
            for node in history:
                # OWNER W W B B
                # TURN  W B W B
                #       + - - +
                r = reward if to_play == player else -reward
                self._backpropagate(node, r)
                to_play = node.to_play
            
            i += 1
            current_time = time.time()
            iter_time = (current_time - start) / i

        # Finally choose the best action at the root according to the policy
        actions, children = zip(*root.children.items())
        # Ensure same ordering of children and actions
        actions = list(actions)
        children = list(children)

        index = self._best_node_policy(children)
        self._save_stats(root, i, time.time() - start)
        return actions[index]

    def update(self, action):
        if not self.reset_tree:
            if action is None:
                self._root = None
            elif self._root is not None:
                if self._root.expanded() and action in self._root.children:
                    self._root = self._root.children[action]
                else:
                    # Reset root in case it was not expanded
                    # It will be expanded in the next call to move
                    self._root = None

    def _save_stats(self, root, iterations, time_spent):
        actions = {
            action: child
            for action, child in root.children.items()
        }

        self._stats = {
            'value': root.value,
            'actions': actions,
            'iters': iterations,
            'time': time_spent
        }

    def _select(self, env, root):
        legal_actions = env.legal_actions()
        # Sort children in the same order as the actions
        children = [root.children[action] for action in legal_actions]
        # Select the child
        index = self._selection_policy(root, children)
        return children[index], legal_actions[index]

    @staticmethod
    def _simulate(env):
        winner = env.winner()
        done = winner is not None
        reward = 0 if not done else winner
        # Simulate until the end of the game
        while not done:
            action = np.random.choice(env.legal_actions())
            _, reward, done, _ = env.step(action)
        return reward

    def _backpropagate(self, node, reward):
        # Translate reward if necessary
        if self._expected_value_range is not None:
            expected_min, expected_max = self._expected_value_range
            observed_min, observed_max = self._observed_value_range
            # We visualize observed range as if observed_min were 0 and
            # observed max 1
            t = (reward - observed_min) / (observed_max - observed_min)
            # Then we set the same proportion in the expected range
            reward = (1 - t) * expected_min + t * expected_max
        node.update(reward, self._update_function)

    @staticmethod
    def _selection_policy_from_string(string):
        string = string.lower()
        if string == 'random':
            return lambda root, children: np.random.choice(len(children))
        elif string == 'uct':
            return UCT(np.sqrt(2))
        elif string == 'omc':
            return OMC()
        elif string == 'pbbm':
            return PBBM()
        return None

    @staticmethod
    def _best_node_policy_from_string(string):
        string = string.lower()
        if string == 'robust':
            return lambda nodes: argmax(nodes, key=lambda n: n.visit_count)
        elif string == 'max':
            return lambda nodes: argmax(nodes, key=lambda n: n.value)
        elif string == 'secure':
            return SecureChild(4)
        return None


class UCT:
    """Class representing the UCT formula (Kocsis and Szepesvári (2006)).

    This formula is used during MCTS' selection phase and chooses a node in:
        argmax_k {v(k) + C * sqrt( Log(n(N)) / n(k) )},
    where N is the parent node, k is in children(N) and C is a constant
    (typically sqrt(2)).

    It is a functional class: (MonteCarloTreeNode, [MonteCarloTreeNode]) -> int.

    """

    value_range = [0, 1]

    def __init__(self, c):
        """

        Args:
            c (float): Exploration constant C.

        """
        self.c = c

    def __call__(self, root, children):
        values = np.array([child.value for child in children])
        visits = np.array([child.visit_count for child in children])
        # Sum 1 to visits to avoid dividing by zero
        uct_values = values + self.c * np.sqrt(
            np.log(root.visit_count) / (visits + 1)
        )
        return np.argmax(uct_values)


class OMC:
    """Class representing the OMC formula (Chaslot et al. (2006a)).

    This formula is used during MCTS' selection phase and chooses a node in:
        argmax_k {(n(N) * U(k)) / (n(k) * sum:{i != k} U(i))},
    where N is the parent node, k is in children(N) and U is the urgency
    function of a node:
        erfc((v0 - v(k)) / (sqrt(2) * std(k))),
    where erfc is the complementary error function and v0 is the highest value
    of all children.

    It is a functional class: (MonteCarloTreeNode, [MonteCarloTreeNode]) -> int.

    """

    value_range = [0, 1]
    _eps = np.finfo(np.float32).eps

    def __call__(self, root, children):
        values = np.array([child.value for child in children])
        visits = np.array([child.visit_count for child in children])
        sigma = np.sqrt([child.value_variance for child in children])
        best_value = values.max()

        value_diff = best_value - values
        out = np.where(value_diff >= 0, np.inf, -np.inf)
        # Divide a / b where sigma is not 0
        # In those cases use +/-np.inf
        # Then compute erfc (erfc(inf = 0))
        urgencies = sp.erfc(np.divide(
            value_diff,
            np.sqrt(2) * sigma,
            where=sigma != 0,
            out=out,
        ))
        # Avoid having all urgencies = 0
        urgencies += self._eps
        urg_sum = urgencies.sum()
        # Sum 1 to visits to avoid dividing by zero
        return np.argmax((root.visit_count * urgencies) /
                         ((visits + 1) * (urg_sum - urgencies)))


class PBBM:
    """Class representing the PBBM formula (Coulom (2006)).

    This formula is used during MCTS' selection phase and chooses a node in:
        argmax_k {(n(N) * U(k)) / (n(k) * sum:{i != k} U(i))},
    where N is the parent node, k is in children(N) and U is the urgency
    function of a node:
        exp(-2.4 * (v0 - v(k)) / (sqrt(2 * (std0^2 + std(k)^2))))
    where v0 is the highest value of all children and std0 the standard
    deviation of the child with the highest value.

    It is a functional class: (MonteCarloTreeNode, [MonteCarloTreeNode]) -> int.

    """

    value_range = [0, 1]
    _eps = np.finfo(np.float32).eps

    def __call__(self, root, children):
        values = np.array([child.value for child in children])
        visits = np.array([child.visit_count for child in children])
        sigma_sq = np.array([child.value_variance for child in children])
        best_node = np.argmax(values)
        best_value = values[best_node]
        best_sigma_sq = sigma_sq[best_node]

        value_diff = -2.4 * (best_value - values)
        sqrt = np.sqrt(2 * (best_sigma_sq + sigma_sq))
        out = np.where(value_diff > 0, np.inf, -np.inf)
        urgencies = np.exp(np.divide(
            value_diff,
            sqrt,
            where=sqrt != 0,
            out=out
        ))
        urgencies += self._eps
        urg_sum = urgencies.sum()
        # Sum 1 to visits to avoid dividing by zero
        return np.argmax((root.visit_count * urgencies) /
                         ((visits + 1) * (urg_sum - urgencies)))


class SecureChild:
    """Class representing the secure child formula.

    It is used at the end of MCTS algorithm to select the returned
    action. Selects the child which maximises a lower confidence bound:
        argmax_k {v(k) + A / n(k)},
    where k is in a set of nodes and A is a constant.

    It is a functional class: ([MonteCarloTreeNode]) -> int.

    """

    def __init__(self, a):
        """

        Args:
            a (float): Confidence bound constant A.

        """
        self.a = a

    def __call__(self, nodes):
        def f(node):
            return node.value + self.a / np.sqrt(node.visit_count)

        return argmax(nodes, key=lambda node: f(node))
