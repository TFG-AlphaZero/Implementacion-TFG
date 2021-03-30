import sys
sys.path.insert(0, '/Documents/Juan Carlos/Estudios/Universidad/5º Carrera/TFG Informatica/ImplementacionTFG')

import numpy as np

from tfg.strategies import Minimax, HumanStrategy
from tfg.util import enable_gpu, play
from tfg.alphaZero import AlphaZero, create_alphazero
from game.tictactoe import TicTacToe, encode, decode
from tfg.debugger import Debugger, NeuralNetworkToPlay
from tfg.alphaZeroAdapters import TicTacToeAdapter

if __name__ == '__main__':
    enable_gpu()

    game = TicTacToe()

    alphaZero = create_alphazero(game, TicTacToeAdapter(),
                                 self_play_times=1, max_games_counter=20,
                                 buffer_size=32, batch_size=16, mcts_iter=100)
    
    alphaZero = AlphaZero(game)
    #alphaZero.load('models/TicTacToe400Iteraciones.h5')
    #alphaZero.train(callbacks=[callback])
    #alphaZero.save('models/TicTacToeParallel.h5')

    debugger = Debugger(alphaZero.neural_network, game)
    #debugger.load("models/Debugger1.h5")
    debugger.train_supervised_nn(max_games_counter=500, epochs=1500)
    debugger.save("models/Debugger1.h5")
    #search_board = np.array([[0, -1, -1],
    #                         [0, 1, 1],
    #                         [0, 0, 1]])
    #search_turn = -1
    #debugger.test_nn(search_board, search_turn)
    #debugger.get_boards()
    #debugger.print_buffer()

    results = play(game, Minimax(game), alphaZero, games=100)
    # print(results)
    #
    # results = play(game, alphaZero, Minimax(game), games=100)
    # print(results)
    #
    # results = play(game, alphaZero, HumanStrategy(game), render=True,
    #                print_results=True)
    #print(results)