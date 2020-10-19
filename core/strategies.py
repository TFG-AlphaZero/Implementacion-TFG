import abc
import copy
import re

import gym
import numpy as np


class Strategy(abc.ABC):

    def move(self, observation):
        raise NotImplementedError


class HumanStrategy(Strategy):

    def __init__(self, env, name='PLAYER'):
        self.env = env
        self.name = str(name)

    def move(self, observation=None):
        self.env.render(mode='human')
        legal_actions = self.env.legal_actions()
        t = self.env.action_space.dtype.type
        print()
        print("[" + self.name + "] ", end="")
        print(f"choose action: {', '.join(map(str, legal_actions))}")
        action = re.split(r'\s+', input("> "))
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            action = t(action)
        else:
            action = np.array(t(a) for a in action)
        return action


class MonteCarloTreeNode:

    def __init__(self, observation):
        self.observation = observation
        self.value = 0
        self.value_sum = 0
        self.visit_count = 0
        self.children = dict()

    def __eq__(self, other):
        return self.observation == other.observation

    def __hash__(self):
        return hash(self.observation)


class MonteCarloTree(Strategy):

    def __init__(self, env, player, max_iter, selection_policy=None, expansion_policy=None, simulation_policy=None,
                 backpropagation_policy=None, best_node_policy=None):
        self._env = env
        self._player = player
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

        # TODO Two actions may lead to the same observation !!
        def _fully_expanded(n):
            return len(n.children) == len(env.legal_actions())

        # Initialize tree
        root = MonteCarloTreeNode(observation)
        root.visit_count += 1

        for _ in range(self.max_iter):
            history = [root]
            # Reset search every iteration
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
                reward *= self._player
                history.append(current_node)

            # We might have found a real leaf node during selection
            if not done:
                # Expansion phase
                parent = current_node
                current_node, action = self._expand(env, parent)
                parent.children[current_node.observation] = current_node
                history.append(current_node)
                env.step(action)

                # Simulation phase
                reward = self._simulate(env) * self._player

            # Backpropagation phase
            for node in reversed(history):
                self._backpropagate(node, reward)

        # Finally choose the best action at the root according to the strategy
        actions = self._env.legal_actions()
        children = []
        for action in actions:
            observation, _, _, _ = self._env.fake_step(action)
            if observation in root.children:
                children.append((action, root.children[observation]))

        # TODO remove
        print("\n".join(str((c.observation, c.visit_count)) for _, c in children))
        return max(children, key=lambda act_node: self._best_node_policy(act_node[1]))[0]

    def _select(self, env, root):
        actions = env.legal_actions()
        children = []
        for action in actions:
            observation, _, _, _ = env.fake_step(action)
            child = root.children[observation]
            children.append(child)
        child = self._selection_policy(root, children)
        index = children.index(child)
        return child, actions[index]

    def _expand(self, env, root):
        actions = env.legal_actions()
        children = []
        for action in actions:
            observation, _, _, _ = env.fake_step(action)
            if observation not in root.children:
                children.append((MonteCarloTreeNode(observation), action))
        return self._expansion_policy(children)

    def _simulate(self, env):
        winner = env.winner()
        done = winner is not None
        reward = 0 if not done else winner
        while not done:
            actions = env.legal_actions()
            act_info = [(action, env.fake_step(action)) for action in actions]
            action = self._simulation_policy(act_info)
            _, reward, done, _ = env.step(action)
        return reward

    def _backpropagate(self, node, reward):
        node.value_sum += reward
        node.visit_count += 1
        node.value = self._backpropagation_policy(node)

    @staticmethod
    def _selection_policy_from_string(string):
        string = string.lower()
        if string == 'random':
            return lambda root, children: np.random.choice(children)
        elif string == 'uct':
            return uct(2)
        return None

    @staticmethod
    def _expansion_policy_from_string(string):
        string = string.lower()
        if string == 'random':
            return lambda children: children[np.random.choice(len(children))]
        return None

    @staticmethod
    def _simulation_policy_from_string(string):
        string = string.lower()
        if string == 'random':
            return lambda act_info: act_info[np.random.choice(len(act_info))][0]
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
    def uctc(root, children):
        values = np.array([child.value_sum / child.visit_count for child in children])
        visits = np.array([child.visit_count for child in children])
        uct_values = values + c * np.sqrt(np.log(root.visit_count) / visits)
        return children[np.argmax(uct_values)]

    return uctc
