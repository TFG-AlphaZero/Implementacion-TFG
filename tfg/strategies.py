import abc
import copy
import random
import re

import gym
import numpy as np

import tfg.games


class Strategy(abc.ABC):
    """
    Interface for game policies. It implements the move method which maps observations to an action.
    """

    def move(self, observation):
        """
        Returns the action that the agent should taken given the current observation. It acts like the policy function.
        :param observation: object - current state of the game
        :return: object - action that the agent should take
        """
        raise NotImplementedError


class HumanStrategy(Strategy):
    """
    Implementation of Strategy so that humans can play a game.
    """

    def __init__(self, env, name='PLAYER'):
        """
        :param env: GameEnv - game this strategy is for
        :param name: String (optional) - name that will be printed for this player (default 'PLAYER')
        """
        self.env = env
        self.name = str(name)

    def move(self, observation=None):
        """
        Renders the game and asks the player to input a move.
        :param observation: ignored as the player will see the current state
        :return: object - action chosen by the human player
        :raise: ValueError if the action is not a valid action in the action space of the environment (the game)
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
            raise ValueError(f"given action {action} is not contained in the action space")
        return action


# TODO docs
class Minimax(Strategy):

    def __init__(self, env, max_depth=None, heuristic=None, alpha_beta=True,
                 ordering=lambda x: random.random()):

        if max_depth is not None and heuristic is None:
            raise ValueError("max_depth was given but heuristic wasn't")

        self._env = env
        self.max_depth = max_depth
        self._heuristic = heuristic
        self.alpha_beta = alpha_beta
        self.ordering = ordering

    def move(self, observation):
        player = self._env.to_play
        action, _ = self._minimax(self._env, observation, player) if not self.alpha_beta else \
            self._alpha_beta(self._env, observation, player)
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

            if selected_action is None or self._better_value(value, reward, player, to_play):
                selected_action = action
                value = reward

        # Return the selected action at the root
        return selected_action, value

    def _alpha_beta(self, env, observation, player, depth=0, alpha=None, beta=None):
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
                    _, reward = self._alpha_beta(env_, obs, player, depth + 1, alpha, beta)

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
                    _, reward = self._alpha_beta(env_, obs, player, depth + 1, alpha, beta)

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
    """
    Node used during the Monte Carlo Tree Search algorithm.
    """

    def __init__(self, observation):
        """
        :param observation: object - state of the game this node is representing
        """
        self.observation = observation
        self.value = 0
        self.value_sum = 0
        self.visit_count = 0
        self.children = dict()

    def __eq__(self, other):
        return self.observation == other.observation

    def __hash__(self):
        return hash(self.observation)


# TODO check if those are the correct papers
class MonteCarloTree(Strategy):
    """
    Game strategy (policy) implementing the Monte Carlo Tree Search algorithm (Coulom, 2006; Kocsis and Szepesvári,
    2006; Chaslot et al., 2006a).

    Essentially, for every move of the player the algorithm will run max_iter iterations of the following:
    1. Selection:       starting in the root traverse down the tree using a given selection_policy until it
                        encounters a node that is not fully expanded.
    2. Expansion:       choose one child of that node using the expansion_policy and add it to the tree.
    3. Simulation:      self-play the rest of the game using the simulation_policy.
    4. Backpropagation: knowing the result of the game (the reward), update the value of every node in the path
                        between the root and the expanded node suing the backpropagation_policy.

    After all that max_iter iterations it selects an action that results in the node that maximizes the value
    returned by best_node_policy.
    """

    def __init__(self, env, max_iter, selection_policy=None, expansion_policy=None, simulation_policy=None,
                 backpropagation_policy=None, best_node_policy=None):
        """
        :param env: GameEnv - game this strategy is for
        :param max_iter: int - total number of iterations of the algorithm for each move
        :param selection_policy: str/function-like - function that will be used to select a child during selection phase
                                 Can be either a function (root: MonteCarloTreeNode, children: [MonteCarloTreeNode]) ->
                                 selected_index: int, or a string representing the function to apply, one of the
                                 following: {'random', 'uct' (UCT: C=sqrt(2))}. Default 'uct'.
        :param expansion_policy: str/function-like - function that will be used to select the expanded node during
                                 expansion phase
                                 Can be either a function (observations: [object], actions: [object)]) ->
                                 selected_index: int, or a string representing the function to apply, one of the
                                 following: {'random'}. Default 'random'.
        :param simulation_policy: str/function-like - function that will be used to select the move that will be
                                  played in each state of the self-play phase
                                  Can be either a function (actions: [object], step_results: [(observation, reward,
                                  done, info)]) -> selected_index: int, or a string  representing the  function to
                                  apply, one of the following: {'random'}. Default 'random'.
        :param backpropagation_policy: str/function-like - function that will be used to update the value of each
                                       node during backpropagation
                                       Can be either a function (node: MonteCarloTreeNode) -> value: number, or a string
                                       representing the function to apply, one of the following: {'mean' (value_sum /
                                       visit_count)}. Default 'mean'.
        :param best_node_policy: str/function-like - function that will be used to select the returned action at the
                                 end of the algorithm
                                 Can be either a function (node: MonteCarloTreeNode) -> value: number (node with
                                 highest value will be chosen), or a string representing the function to apply,
                                 one of the following: {'count' (visit_count)}. Default 'count'.
        """

        self._env = env
        self.max_iter = max_iter

        self._selection_policy = self._selection_policy_from_string(selection_policy) \
            if isinstance(selection_policy, str) else \
            selection_policy or self._selection_policy_from_string('uct')

        self._expansion_policy = self._expansion_policy_from_string(expansion_policy) \
            if isinstance(expansion_policy, str) else \
            expansion_policy or self._expansion_policy_from_string('random')

        self._simulation_policy = self._simulation_policy_from_string(simulation_policy) \
            if isinstance(simulation_policy, str) else \
            simulation_policy or self._simulation_policy_from_string('random')

        self._backpropagation_policy = self._backpropagation_policy_from_string(backpropagation_policy) \
            if isinstance(backpropagation_policy, str) else \
            backpropagation_policy or self._backpropagation_policy_from_string('mean')

        self._best_node_policy = self._best_node_policy_from_string(best_node_policy) \
            if isinstance(best_node_policy, str) else \
            best_node_policy or self._best_node_policy_from_string('count')

    @property
    def env(self):
        return self._env

    def move(self, observation):
        """
        Implementation of the MCTS algorithm.
        :param observation: object - current state of the game (must be the same as the one in self.env or the
                            algorithm might not work)
        :return: object - action that the agent should take
        """

        # TODO Two actions may lead to the same observation !!
        def _fully_expanded(n):
            return len(n.children) == len(env.legal_actions())

        player = self._env.to_play

        # Initialize tree
        root = MonteCarloTreeNode(observation)
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
                # Update reward: if WHITE won reward=1, but if we are black reward should be -1 as we lost
                reward *= player
                history.append(current_node)

            # We might have found a real leaf node during selection so there is no need to expand or simulate
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
            for node in reversed(history):
                self._backpropagate(node, reward)

        # Finally choose the best action at the root according to the policy
        actions = self._env.legal_actions()
        children = []
        for action in actions:
            observation, _, _, _ = self._env.step(action, fake=True)
            if observation in root.children:
                children.append((action, root.children[observation]))

        # TODO remove
        # print("\n".join(str((c.observation, c.visit_count)) for _, c in children))
        return max(children, key=lambda act_node: self._best_node_policy(act_node[1]))[0]

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
        # Filter non visited states
        for action in legal_actions:
            observation, _, _, _ = env.step(action, fake=True)
            if observation not in root.children:
                observations.append(observation)
                actions.append(action)
        # Select the child (and the action)
        index = self._expansion_policy(observations, actions)
        observation = observations[index]
        action = actions[index]
        return MonteCarloTreeNode(observation), action

    def _simulate(self, env):
        winner = env.winner()
        done = winner is not None
        reward = 0 if not done else winner
        # Simulate until the end of the game
        while not done:
            legal_actions = env.legal_actions()
            step_results = [env.step(action, fake=True) for action in legal_actions]
            index = self._simulation_policy(legal_actions, step_results)
            action = legal_actions[index]
            # Take the actual step
            _, reward, done, _ = env.step(action)
        return reward

    # TODO check if it matters whose turn is in a node
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
            return uct(np.sqrt(2))
        return None

    @staticmethod
    def _expansion_policy_from_string(string):
        string = string.lower()
        if string == 'random':
            return lambda observations, actions: np.random.choice(len(observations))
        return None

    @staticmethod
    def _simulation_policy_from_string(string):
        string = string.lower()
        if string == 'random':
            return lambda actions, step_results: np.random.choice(len(actions))
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


def uct(c):
    """
    Returns the function that implements the UCT formula to select a node in the selection phase of the MCTS algorithm:
        argmax_k {v(k) + C * sqrt( Log(n(N)) / n(k) )},
    where N is the parent node, k is in children(N) and C is a constant (typically sqrt(2)).
    :param c: float - exploration constant C
    :return: function (MonteCarloTreeNode, [MonteCarloTreeNode]) -> MonteCarloTreeNode - the actual UCT function
    """

    def uctc(root, children):
        values = np.array([child.value for child in children])
        visits = np.array([child.visit_count for child in children])
        uct_values = values + c * np.sqrt(np.log(root.visit_count) / visits)
        return np.argmax(uct_values)

    return uctc
