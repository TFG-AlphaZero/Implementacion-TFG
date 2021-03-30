import numpy as np
import tfg.alphaZeroConfig as config

from strategies import Strategy, Minimax
from tfg.alphaZero import AlphaZero
from matplotlib import pyplot as plt

class Debugger(Strategy):

    def __init__(self, neural_network, env, alphaZero = None):
        self.neural_network = neural_network
        self.alphaZero = alphaZero
        self._env = env
        self.solver = Minimax(env)

        np.set_printoptions(suppress=True)
        np.set_printoptions(precision=2)

    def move(self, observation):
        player = self._env.to_play
        legal_actions = observation.flatten()
        
        nn_input = np.array([AlphaZero._convert_to_network_input(observation, player)])
        predictions = self.neural_network.predict(nn_input)
        
        reward = predictions[0][0][0]
        probabilities = predictions[1][0]
        
        #print(reward, probabilities)
        
        probabilities[legal_actions != 0] = 0
        return np.argmax(probabilities)

    def save(self, path):
        self.neural_network.save_model(path)

    def load(self, path):
        self.neural_network.load_model(path)

    def plot_history(self, history):
        fig, (ax1, ax2) = plt.subplots(1, 2)

        ax1.plot(history.history['policy_head_loss'])
        ax1.plot(history.history['val_policy_head_loss'])
        ax1.set_title("Model Loss")
        ax1.set_ylabel('Loss')
        ax1.set_xlabel('Epoch')
        ax1.legend(['Train', 'Validation'], loc='upper left')

        ax2.plot(history.history['policy_head_acc'])
        ax2.plot(history.history['val_policy_head_acc'])
        ax2.set_title('Model Accuracy')
        ax2.set_ylabel('Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.legend(['Train', 'Validation'], loc='upper left')

        plt.show()
        fig.savefig('models/DebugLossAcc.png')

    def train_supervised_nn(self, max_games_counter=config.MAX_GAMES_COUNTER, epochs=config.EPOCHS):        
        def _self_play(max_games_counter):
            game_buffer = []

            for aux in range(max_games_counter):
                observation = self._env.reset()
                game_data = []

                while True:
                    action = self.solver.move(observation)
                    pi = np.zeros(9)
                    pi[action] = 1
                    game_data.append((observation, self._env.to_play, pi))
                    observation, _, done, _ = self._env.step(action)

                    if done:
                        break

                for i in range(len(game_data)):
                    game_data[i] += (self._env.winner(),)
                
                game_buffer.extend(game_data)
                print(aux)

            return game_buffer

        moves = _self_play(max_games_counter)

        # Separate data from batch
        boards, turns, pies, rewards = zip(*moves)
        train_board = np.array([
            AlphaZero._convert_to_network_input(board, turn)
            for board, turn in zip(boards, turns)]
        )
        train_pi = np.array(list(pies))
        train_reward = np.array(list(rewards))

        # Train neural network with the data from the mini-batch
        history = self.neural_network.fit(x=train_board,
                                          y=[train_reward, train_pi],
                                          batch_size=32,
                                          epochs=epochs,
                                          verbose=0,
                                          validation_split=0.1,
                                          shuffle = True)
        self.plot_history(history)
        

    def test_nn(self, search_board, search_turn):
        print("================ \n Search Board: \n", search_board)
        X = np.array([AlphaZero._convert_to_network_input(search_board, search_turn)])
        print("\n Neural Network Input: \n", X)

        predictions = self.neural_network.predict(x = X)
        print("\nPredicted reward: ", predictions[0][0][0], '\nPredicted probabilities: ', predictions[1][0])

    def get_boards(self):
        search_board = np.array([[0, 0, -1],
                                 [0, 1, 0],
                                 [0, 0, 0]])
        search_turn = 1

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
