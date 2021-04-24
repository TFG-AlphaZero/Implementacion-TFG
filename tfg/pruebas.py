import sys
sys.path.insert(0, '/Documents/Juan Carlos/Estudios/Universidad/5º Carrera/TFG Informatica/ImplementacionTFG')

import numpy as np
import time

from tfg.strategies import Minimax, HumanStrategy, MonteCarloTree
from tfg.util import enable_gpu, play
from tfg.alphaZero import AlphaZero, create_alphazero, parallel_play
from game.tictactoe import TicTacToe, encode, decode
from game.connect_n import ConnectN
from tfg.debugger import Debugger
from tfg.alphaZeroAdapters import TicTacToeAdapter, ConnectNAdapter

if __name__ == '__main__':
    enable_gpu()

    game = ConnectN()
    # print(parallel_play(game, TicTacToeAdapter(), Minimax(game),
    #                     '../experiments/models/experiment_tictactoe.h5',
    #                     'black', max_workers=10, mcts_iter=400))

    #alphaZero = create_alphazero(game, ConnectNAdapter(game),
    #                             self_play_times=10, max_games_counter=200,
    #                             buffer_size=1500, batch_size=512, 
    #                             mcts_iter=100, epochs=20,
    #                             exploration_noise=(.25, .045), c_puct=1,
    #                             max_workers=5)
    
    alphaZero = AlphaZero(game, adapter=ConnectNAdapter(game), mcts_iter=100)
    #alphaZero.load('models/ConnectN_v1.h5')
    alphaZero.train(self_play_times=10, max_games_counter=100)
    #alphaZero.save('models/ConnectN_v1.h5')

    #debugger = Debugger(game, adapter=TicTacToeAdapter(), nn=alphaZero.neural_network)
    #debugger.load("models/Debugger2.h5")
    #history = debugger.train_supervised_nn(max_games_counter=100, epochs=250)
    #debugger.plot_history(history, "models/Debugger2Loss")
    #debugger.save("models/Debugger2.h5")
    #debugger.test_nn()

    #Play as black against Minimax
    #results = play(game, alphaZero, MonteCarloTree(game, max_iter=400), games=5)
    #print(results)
    
    #Play as white against Minimax
    #results = play(game, debugger, Minimax(game), games=10)
    #print(results)
    
    #Play against human
    #results = play(game, HumanStrategy(game), alphaZero, render=True,
    #               print_results=True)
    #print(results)