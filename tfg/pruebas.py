import sys
sys.path.insert(0, '/Documents/Juan Carlos/Estudios/Universidad/5º Carrera/TFG Informatica/ImplementacionTFG')

from tfg.strategies import Minimax, HumanStrategy
from tfg.util import enable_gpu, play
from tfg.alphaZero import AlphaZero, create_alphazero
from game.tictactoe import TicTacToe, encode, decode
from tfg.debugger import Debugger, NeuralNetworkToPlay
from tfg.alphaZeroCallbacks import PositionCheckerCallback

if __name__ == '__main__':
    enable_gpu()

    game = TicTacToe()

    callback = PositionCheckerCallback(encode)

    alphaZero = create_alphazero(game, max_workers=8, self_play_times=32,
                                 max_games_counter=1024, buffer_size=512,
                                 callbacks=[callback], mcts_times=400)
    
    #alphaZero = AlphaZero(game)
    #alphaZero.load('models/TicTacToe400Iteraciones.h5')
    #alphaZero.train()
    alphaZero.save('models/TicTacToeParallel.h5')

    #debugger = Debugger(alphaZero)
    #debugger.test_nn()
    #debugger.get_boards()
    #debugger.print_buffer()
    #nn_play = NeuralNetworkToPlay(game, alphaZero.neural_network)

    results = play(game, Minimax(game), alphaZero, games=100)
    print(results)

    results = play(game, alphaZero, Minimax(game), games=100)
    print(results)

    results = play(game, alphaZero, HumanStrategy(game), render=True,
                   print_results=True)
    #print(results)