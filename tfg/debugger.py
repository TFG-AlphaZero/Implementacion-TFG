import numpy as np
from strategies import Strategy

class Debugger():

    def __init__(self, alphaZero):
        self.alphaZero = alphaZero

    def test_nn(self):
        search_board = np.array([[0, 0, -1],
                                 [0, 1, 0],
                                 [0, 0, 0]])
        search_play = 1
        
        X = np.array([AlphaZero._convert_to_network_input(search_board, search_turn)])
        print("================ \n Neural Network Input: \n", X)

        predictions = alphaZero.neural_network.predict(x = X)
        print("\nPredicted reward: ", predictions[0][0][0], '\nPredicted probabilities: ', predictions[1][0])

    def get_boards(self):
        search_board = np.array([[0, 0, -1],
                                 [0, 1, 0],
                                 [0, 0, 0]])
        search_play = 1

        buffer = self.alphaZero._buffer
        buffer_board = []

        for elem in buffer :
            board = elem[0][0]
            turn = elem[0][1]
            if (board == search_board).all() and search_turn == turn :
                buffer_board.append(elem)

        print(buffer_board)

    def print_buffer(self):
        buffer = self.alphaZero._buffer
        mapping = {-1: 'O', 0: ' ', 1: 'X'}        

        for elem in buffer :
            board = elem[0][0]
            turn = elem[0][1]
            pi = elem[1]
            winner = elem[2]

            #Print board
            print("=================")
            tokens = [[mapping[cell] for cell in row] for row in board]
            
            print("\n-+-+-\n".join(
                ['|'.join([token for token in row]) for row in tokens]
            )+'\n')

            #Print turn, pi vector and winner
            aux = "(Blancas X)" if turn == 1 else "(Negras O)"
            print("Turno: ", turn, aux)
            print("Vector Pi: ", pi)
            print("Winner: ", winner)
            
class NeuralNetworkToPlay(Strategy):
    
    def __init__(self, env, neural_network):
        self.neural_network = neural_network
        self.env = env

    def move(self, observation):
        player = self.env.to_play
        legal_actions = observation.flatten()
        
        nn_input = np.array([self._convert_to_network_input(observation, player)])
        predictions = self.neural_network.predict(nn_input)
        
        reward = predictions[0][0][0]
        probabilities = predictions[1][0]
        
        np.set_printoptions(suppress=True)
        np.set_printoptions(precision=2)
        print(reward, probabilities)
        
        probabilities[legal_actions != 0] = 0
        return np.argmax(probabilities)

    def _convert_to_network_input(self, board, to_play):
        black = board.copy()
        black[black == 1] = 0
        black[black == -1] = 1
        white = board.copy()
        white[white == -1] = 0
        player = 0 if to_play == 1 else 1
        turn = np.full(board.shape, player)
        input = np.stack((black, white, turn), axis = 2)
        return input