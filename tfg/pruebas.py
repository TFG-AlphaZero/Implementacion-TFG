import sys
sys.path.insert(0, '/Documents/Juan Carlos/Estudios/Universidad/5º Carrera/TFG Informatica/ImplementacionTFG')

import numpy as np

from tfg.strategies import Minimax, HumanStrategy
from tfg.util import enable_gpu, play
from tfg.alphaZero import AlphaZero, create_alphazero
from game.tictactoe import TicTacToe, encode, decode
from tfg.debugger import Debugger
from tfg.alphaZeroAdapters import TicTacToeAdapter

if __name__ == '__main__':
    enable_gpu()

    game = TicTacToe()

    #alphaZero = create_alphazero(game, TicTacToeAdapter(),
    #                             self_play_times=1, max_games_counter=20,
    #                             buffer_size=32, batch_size=16, mcts_iter=100)
    
    #alphaZero = AlphaZero(game, adapter=TicTacToeAdapter())
    #alphaZero.load('models/TicTacToe400Iteraciones.h5')
    #alphaZero.train(callbacks=[callback])
    #alphaZero.save('models/TicTacToeParallel.h5')

    #debugger = Debugger(game, adapter=TicTacToeAdapter())
    #debugger.load("models/Debugger2.h5")
    #history = debugger.train_supervised_nn(max_games_counter=100, epochs=250)
    #debugger.plot_history(history, "models/Debugger2Loss")
    #debugger.save("models/Debugger2.h5")
    #debugger.test_nn()

    #Play as black against Minimax
    #results = play(game, Minimax(game), debugger, games=10)
    #print(results)
    
    #Play as white against Minimax
    #results = play(game, debugger, Minimax(game), games=10)
    #print(results)
    
    #Play against human
    #results = play(game, HumanStrategy(game), debugger, render=True,
    #               print_results=True)
    #print(results)