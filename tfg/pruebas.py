import sys
sys.path.insert(0, '/Documents/Juan Carlos/Estudios/Universidad/5º Carrera/TFG Informatica/ImplementacionTFG')

import numpy as np

from tfg.strategies import Minimax, HumanStrategy
from tfg.util import enable_gpu, play
from tfg.alphaZero import AlphaZero, create_alphazero
from game.tictactoe import TicTacToe
from tfg.debugger import Debugger, NeuralNetworkToPlay

if __name__ == '__main__':
    #enable_gpu()

    game = TicTacToe()

    #alphaZero = create_alphazero(game, max_workers=4, self_play_times=8,
    #                             max_train_time=10)
    
    #alphaZero = AlphaZero(game)
    #alphaZero.load('models/TicTacToe400Iteraciones.h5')
    #alphaZero.train()
    #alphaZero.save('models/TicTacToe400Iteraciones.h5')

    #debugger = Debugger(alphaZero)
    #debugger.test_nn()
    #debugger.get_boards()
    #debugger.print_buffer()
    #nn_play = NeuralNetworkToPlay(game, alphaZero.neural_network)

    #results = play(game, Minimax(game), alphaZero, games=100)
    #print(results)

    #results = play(game, alphaZero, Minimax(game), games=100)
    #print(results)

    #results = play(game, alphaZero, HumanStrategy(game), games = 1)    
    #print(results)