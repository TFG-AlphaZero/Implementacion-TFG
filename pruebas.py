import sys
sys.path.insert(0, '/Documents/Juan Carlos/Estudios/Universidad/5º Carrera/TFG Informatica/ImplementacionTFG')

import numpy as np
from gym.spaces import Discrete, Box

from tfg.games import GameEnv, WHITE, BLACK
from tfg.strategies import MonteCarloTree, HumanStrategy, Minimax
from tfg.util import play
from tfg.alphaZero import AlphaZero
from examples.connect_n import ConnectN
from examples.tictactoe import TicTacToe

game = TicTacToe()
alphaZero = AlphaZero(game, self_play_times=1)
buffer = alphaZero.train()
print(buffer)
