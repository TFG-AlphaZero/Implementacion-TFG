import abc
import copy
import random
import re

import gym
import numpy as np

import tfg.games


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
            max_depth (:obj:`int`, optional): Maximum depth the tree is allowed
                to grow. If None it will grow until a
                leaf node is reached. If set, heuristic must be given. Defaults
                    to None.
            heuristic (function, optional): If max_depth is not None, heuristic
                function (observation: object, to_play: int) -> int that will be
                called to estimate the value of a leaf node. This value will
                be positive if WHITE is more likely to win, negative if BLACK is
                the one who is winning and 0 if the game is estimated to end
                in a draw. It is recommended that the return value of this
                function is between -1 and 1.
            alpha_beta (:obj:`bool`, optional): Determines whether or not to use
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


class MonteCarloTreeNode:
    """Node used during the Monte Carlo Tree Search algorithm."""

    def __init__(self, observation, to_play):
        """

        Args:
            observation (object): State of the game this node is representing.
            to_play (:obj:`int`): Player to play in this state, either BLACK
                or WHITE.

        """
        self.observation = observation
        self.to_play = to_play
        self.value = 0
        self.value_sum = 0
        self.visit_count = 0
        self.children = dict()

    def __eq__(self, other):
        return self.observation == other.observation and self.to_play == other.to_play

    def __hash__(self):
        return hash((self.observation, self.to_play))


# TODO check if those are the correct papers
class MonteCarloTree(Strategy):
    """Game strategy (policy) implementing Monte Carlo Tree Search algorithm

    This algorithm was first described in Coulom, 2006; Kocsis and Szepesvári,
    2006; and Chaslot et al., 2006a.

    Essentially, for every move of the player the algorithm will run max_iter
    iterations of the following:
    1. Selection:       starting in the root traverse down the tree using a
                        given selection_policy until it encounters a node
                        that is not fully expanded.
    2. Expansion:       choose one child of that node using the expansion_policy
                        and add it to the tree.
    3. Simulation:      self-play the rest of the game using the
                        simulation_policy.
    4. Backpropagation: knowing the result of the game (the reward), update the
                        value of every node in the path between the root and
                        the expanded node suing the backpropagation_policy.

    After all that max_iter iterations it selects an action that results in the
    node that maximizes the value returned by best_node_policy.

    """

    def __init__(self, env, max_iter,
                 selection_policy=None,
                 expansion_policy=None,
                 simulation_policy=None,
                 backpropagation_policy=None,
                 best_node_policy=None):
        """

        Args:
            env (tfg.games.GameEnv): Game this strategy is for.
            max_iter (:obj:`int`): Total number of iterations of the algorithm
                for each move.
            selection_policy (function or :obj:`str`, optional): Function
                that will be used to select a child during selection phase Can
                be either a function
                    (root: MonteCarloTreeNode,
                    children: [MonteCarloTreeNode]) -> selected_index: int,
                or a string representing the function to apply, one of the
                following: {'random', 'uct'}, where 'uct' uses C=sqrt(2).
                Defaults to 'uct'.
            expansion_policy (function or :obj:`str`, optional): Function that
                will be used to select the expanded node during expansion
                phase. Can be either a function
                    (observations: [object],
                    actions: [object)]) -> selected_index: int,
                or a string representing the function to apply, one of the
                following: {'random'}. Defaults to 'random'.
            simulation_policy (function or :obj:`str`, optional): Function that
                will be used to select the move that will be played in each
                state of the self-play phase. Can be either a function
                    (actions: [object],
                    step_results: [(observation, reward, done, info)]) ->
                    selected_index: int,
                or a string  representing the  function to apply, one of the
                following: {'random'}. Defaults to 'random'.
            backpropagation_policy (function or :obj:`str`, optional): Function
                that will be used to update the value of each node during
                backpropagation. Can be either a function
                    (node: MonteCarloTreeNode) -> value: number,
                or a string representing the function to apply, one of the
                following: {'mean'},  where 'mean' is computed over visit
                count. Defaults to 'mean'.
            best_node_policy (function or :obj:`str`, optional): Function that
                will be used to select the returned action at the end of the
                algorithm. Can be either a function
                    (node: MonteCarloTreeNode) -> value: number
                (node with highest value will be chosen), or a string
                representing the function to apply, one of the following:
                {'count'}, where 'count' returns the visit count of the node.
                Defaults to 'count'.

        """

        if selection_policy is None:
            selection_policy = 'uct'

        if expansion_policy is None:
            expansion_policy = 'random'

        if simulation_policy is None:
            simulation_policy = 'random'

        if backpropagation_policy is None:
            backpropagation_policy = 'mean'

        if best_node_policy is None:
            best_node_policy = 'count'

        self._env = env
        self.max_iter = max_iter

        self._selection_policy = (
            self._selection_policy_from_string(selection_policy)
            if isinstance(selection_policy, str)
            else selection_policy
        )

        self._expansion_policy = (
            self._expansion_policy_from_string(expansion_policy)
            if isinstance(expansion_policy, str)
            else expansion_policy
        )

        self._simulation_policy = (
            self._simulation_policy_from_string(simulation_policy)
            if isinstance(simulation_policy, str)
            else simulation_policy
        )

        self._backpropagation_policy = (
            self._backpropagation_policy_from_string(backpropagation_policy)
            if isinstance(backpropagation_policy, str)
            else backpropagation_policy
        )

        self._best_node_policy = (
            self._best_node_policy_from_string(best_node_policy)
            if isinstance(best_node_policy, str)
            else best_node_policy
        )

    @property
    def env(self):
        """tfg.games.GameEnv: Game this strategy is for."""
        return self._env

    def move(self, observation):
        """Implementation of the MCTS algorithm.

        Args:
            observation (object): Current state of the game (must be the same
                as the one in self.env or the algorithm might not work).

        Returns:
            object: Action that the agent should take.

        """

        # TODO Two actions may lead to the same observation !!
        def _fully_expanded(n):
            return len(n.children) == len(env.legal_actions())

        player = self._env.to_play

        # Initialize tree
        root = MonteCarloTreeNode(observation, player)
        root.visit_count += 1

        # Iterate the algorithm max_iter times
        for _ in range(self.max_iter):
            # Reset search every iteration
            history = [root]
            env = copy.deepcopy(self._env)
            done = False
            reward = 0

            current_node = root

            # Selection phase
            while _fully_expanded(current_node) and not done:
                # Select next child
                current_node, action = self._select(env, current_node)

                # Take the action
                _, reward, done, _ = env.step(action)
                # Update reward: if WHITE won reward=1,
                # but if we are black reward should be -1 as we lost
                reward *= player
                history.append(current_node)

            # We might have found a real leaf node during selection
            # so there is no need to expand or simulate
            if not done:
                # Expansion phase
                parent = current_node
                current_node, action = self._expand(env, parent)
                # Add expanded node to parent
                parent.children[current_node.observation] = current_node
                history.append(current_node)
                env.step(action)

                # Simulation phase
                reward = self._simulate(env) * player

            # Backpropagation phase
            # Who played the move that lead to that node
            to_play = player
            for node in history:
                # OWNER W W B B
                # TURN  W B W B
                #       + - - +
                r = reward if to_play == player else -reward
                self._backpropagate(node, r)
                to_play = node.to_play

        # Finally choose the best action at the root according to the policy
        legal_actions = self._env.legal_actions()
        children = []
        actions = []
        for action in legal_actions:
            observation, _, _, _ = self._env.step(action, fake=True)
            if observation in root.children:
                children.append(root.children[observation])
                actions.append(action)

        index = max(range(len(children)),
                    key=lambda i: self._best_node_policy(children[i]))
        return actions[index]

    def _select(self, env, root):
        legal_actions = env.legal_actions()
        children = []
        # Sort children in the same order as the actions
        for action in legal_actions:
            observation, _, _, _ = env.step(action, fake=True)
            children.append(root.children[observation])
        # Select the child
        index = self._selection_policy(root, children)
        return children[index], legal_actions[index]

    def _expand(self, env, root):
        legal_actions = env.legal_actions()
        observations = []
        actions = []
        to_play = []
        # Filter non visited states
        for action in legal_actions:
            # We need to take the step so we can now whose turn is now
            env_ = copy.deepcopy(env)
            observation, _, _, _ = env_.step(action)
            if observation not in root.children:
                observations.append(observation)
                actions.append(action)
                to_play.append(env_.to_play)
        # Select the child (and the action)
        index = self._expansion_policy(observations, actions)
        observation = observations[index]
        action = actions[index]
        turn = to_play[index]
        return MonteCarloTreeNode(observation, turn), action

    def _simulate(self, env):
        winner = env.winner()
        done = winner is not None
        reward = 0 if not done else winner
        # Simulate until the end of the game
        while not done:
            legal_actions = env.legal_actions()
            step_results = [env.step(action, fake=True)
                            for action in legal_actions]
            index = self._simulation_policy(legal_actions, step_results)
            action = legal_actions[index]
            # Take the actual step
            _, reward, done, _ = env.step(action)
        return reward

    def _backpropagate(self, node, reward):
        # Use reward in [0, 1], where loss=0, draw=.5, win=1
        reward = (reward + 1) / 2
        # First update all basic fields of the node
        node.value_sum += reward
        node.visit_count += 1
        # Then update its value
        node.value = self._backpropagation_policy(node)

    @staticmethod
    def _selection_policy_from_string(string):
        string = string.lower()
        if string == 'random':
            return lambda root, children: np.random.choice(len(children))
        elif string == 'uct':
            return UCT(np.sqrt(2))
        return None

    @staticmethod
    def _expansion_policy_from_string(string):
        string = string.lower()
        if string == 'random':
            return lambda obs, acts: np.random.choice(len(obs))
        return None

    @staticmethod
    def _simulation_policy_from_string(string):
        string = string.lower()
        if string == 'random':
            return lambda acts, step_results: np.random.choice(len(acts))
        return None

    @staticmethod
    def _backpropagation_policy_from_string(string):
        string = string.lower()
        if string == 'mean':
            return lambda node: node.value_sum / node.visit_count
        return None

    @staticmethod
    def _best_node_policy_from_string(string):
        string = string.lower()
        if string == 'count':
            return lambda node: node.visit_count
        return None


class UCT:
    """Class representing the UCT formula.

    This formula is used during MCTS' selection phase and chooses a node in:
        argmax_k {v(k) + C * sqrt( Log(n(N)) / n(k) )},
    where N is the parent node, k is in children(N) and C is a constant
    (typically sqrt(2)).

    It is a functional class: (MonteCarloTreeNode, [MonteCarloTreeNode]) -> int.

    """

    def __init__(self, c):
        """

        Args:
            c (float): Exploration constant C.

        """
        self.c = c

    def __call__(self, root, children):
        values = np.array([child.value for child in children])
        visits = np.array([child.visit_count for child in children])
        uct_values = values + self.c * np.sqrt(
            np.log(root.visit_count) / visits
        )
        return np.argmax(uct_values)
