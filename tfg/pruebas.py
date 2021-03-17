import sys
sys.path.insert(0, '/Documents/Juan Carlos/Estudios/Universidad/5º Carrera/TFG Informatica/ImplementacionTFG')

import tensorflow as tf
import numpy as np

from tfg.strategies import Minimax, HumanStrategy
from tfg.util import play
from tfg.alphaZero import AlphaZero
from game.tictactoe import TicTacToe

# GPU didn't work otherwise
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

class DebuggerAZ():

    def __init__(self, alphaZero):
        self.alphaZero = alphaZero

    def test_nn(self, search_board, search_turn):
        X = np.array([self.alphaZero._convert_to_network_input(search_board, search_turn)])
        print(X)
        
        predictions = alphaZero.neural_network.predict(x = X)
        print(predictions[0][0][0].numpy(), '\n', predictions[1][0].numpy())

    def get_boards(self, search_board, search_turn):
        buffer = self.alphaZero._buffer
        buffer_board = []

        for elem in buffer :
            board = elem[0][0]
            turn = elem[0][1]
            if (board == search_board).all() and search_turn == turn :
                buffer_board.append(elem)

        print(buffer_board)


if __name__ == '__main__':
    game = TicTacToe()

    alphaZero = AlphaZero(game)
    #alphaZero.load('models/TicTacToeDemoNN.h5')
    alphaZero.train()
    #alphaZero.save('models/TicTacToeDemoNN.h5')

    #play(game, HumanStrategy(game), alphaZero, render=True, print_results=True)

    debugger = DebuggerAZ(alphaZero)
    search_board = np.array([[1,-1, -1],
                             [0, 1, 0],
                             [0, 0, 0]])
    search_play = 1
    debugger.test_nn(search_board, search_play)
    #debugger.get_boards(search_board, search_play)
    
    #results = play(game, Minimax(game), alphaZero, games=10)
    #print(results)
    
    #results = play(game, alphaZero, Minimax(game), games = 10)
    #print(results)
