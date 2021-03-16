import sys
sys.path.insert(0, '/Documents/Juan Carlos/Estudios/Universidad/5º Carrera/TFG Informatica/ImplementacionTFG')

import tensorflow as tf

from tfg.strategies import Minimax
from tfg.util import play
from tfg.alphaZero import AlphaZero
from examples.tictactoe import TicTacToe


# GPU didn't work otherwise
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

if __name__ == '__main__':
    game = TicTacToe()

    alphaZero = AlphaZero(game)
    #alphaZero.load('models/TicTacToeDemo.h5')
    #alphaZero.train()
    #alphaZero.save('models/TicTacToeDemoV2.h5')

    play(game, HumanStrategy(game), alphaZero, render=True, print_results=True)

    #results = play(game, Minimax(game), alphaZero, games=10)
    #print(results)
    
    #results = play(game, alphaZero, Minimax(game), games = 10)
    #print(results)
