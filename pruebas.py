import sys
sys.path.insert(0, '/Documents/Juan Carlos/Estudios/Universidad/5º Carrera/TFG Informatica/ImplementacionTFG')

import numpy as np
from gym.spaces import Discrete, Box

from tfg.games import GameEnv, WHITE, BLACK
from tfg.strategies import MonteCarloTree, HumanStrategy, Minimax, MonteCarloTreeNode
from tfg.util import play
from tfg.alphaZero import AlphaZero
from examples.connect_n import ConnectN
from examples.tictactoe import TicTacToe


if __name__ == '__main__':
    game = TicTacToe()
    alphaZero = AlphaZero(game, self_play_times=3)
    # alphaZero.load('models/TicTacToeDemo.h5')
    alphaZero.train()
    alphaZero.save('models/TicTacToeDemo.h5')
    play(game, Minimax(game), alphaZero, render=True, print_results=True)
    results = play(game, Minimax(game), alphaZero, games=10)
    print(results)
