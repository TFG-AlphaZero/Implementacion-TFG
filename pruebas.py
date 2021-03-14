import sys
sys.path.insert(0, '/Documents/Juan Carlos/Estudios/Universidad/5º Carrera/TFG Informatica/ImplementacionTFG')

import numpy as np
import tfg.alphaZeroConfig as config

from gym.spaces import Discrete, Box
from tfg.games import GameEnv, WHITE, BLACK
from tfg.strategies import MonteCarloTree, HumanStrategy, Minimax, MonteCarloTreeNode
from tfg.util import play
from tfg.alphaZero import AlphaZero
from examples.connect_n import ConnectN
from examples.tictactoe import TicTacToe


if __name__ == '__main__':
    game = TicTacToe()

    alphaZero = AlphaZero(game)
    alphaZero.load('models/TicTacToeDemo.h5')
    alphaZero.train()
    alphaZero.save('models/TicTacToeDemo.h5')

    results = play(game, Minimax(game), alphaZero, games = 10, max_workers = config.MAX_WORKERS)
    print(results)
    #results = play(game, alphaZero, Minimax(game), games = 15)
    #print(results)
