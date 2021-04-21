import numpy as np
import tfg.alphaZeroConfig as config

from tfg.strategies import Strategy, Minimax
from tfg.alphaZero import AlphaZero
from tfg.alphaZeroNN import NeuralNetworkAZ
from matplotlib import pyplot as plt
from tfg.util import enable_gpu, play
import time

class Debugger(Strategy):

    def __init__(self, env, adapter, nn = None):
        self._env = env
        self.solver = Minimax(env)
        self.adapter = adapter

        if nn is None :
            #Initialize neural network
            nn_config = config.AlphaZeroConfig()
            nn = NeuralNetworkAZ(
                input_dim=adapter.input_shape,
                output_dim=adapter.output_features,
                **nn_config.__dict__
            )
        self.neural_network = nn

        #Format pi vector properly
        np.set_printoptions(suppress=True)
        np.set_printoptions(precision=2)

    def move(self, observation):
        player = self._env.to_play
        legal_actions = observation.flatten()
        
        nn_input = np.array([self.adapter.to_input(observation, player)])
        predictions = self.neural_network.predict(nn_input)
        
        reward = predictions[0][0][0]
        probabilities = predictions[1][0]
        
        #print(observation)
        #print(reward, probabilities)
        
        probabilities[legal_actions != 0] = 0
        return np.argmax(probabilities)

    def save(self, path):
        self.neural_network.save_model(path)

    def load(self, path):
        self.neural_network.load_model(path)

    def plot(self, path):
        self.neural_network.plot_model(path)
    
    def summary(self):
        self.neural_network.model.summary()

    def plot_history(self, history, path):
        plt.plot(history.history['policy_head_loss'])
        plt.plot(history.history['val_policy_head_loss'])
        plt.title("Model Loss")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.ylim(0, 10)
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.savefig(path)

    def _self_play(self, max_games_counter):
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

    def get_train_data(self, moves) :
        # Separate data from batch
        boards, turns, pies, rewards = zip(*moves)
        train_board = np.array([
            self.adapter.to_input(board, turn)
            for board, turn in zip(boards, turns)]
        )
        train_pi = np.array(list(pies))
        train_reward = np.array(list(rewards))

        return train_board, train_pi, train_reward

    def train_supervised_nn(self, max_games_counter=config.MAX_GAMES_COUNTER, epochs=config.EPOCHS):        
        moves = self._self_play(max_games_counter)

        train_board, train_pi, train_reward = self.get_train_data(moves)

        # Train neural network with the data from the mini-batch
        history = self.neural_network.fit(x=train_board,
                                          y=[train_reward, train_pi],
                                          batch_size=32,
                                          epochs=epochs,
                                          verbose=2,
                                          validation_split=0.1,
                                          shuffle = True)
        return history
        
    def test_nn(self, moves = None):
        if moves is None :
            #Play a game against solver
            while True :
                moves = self._self_play(1)
                if moves[-1][-1] != 0 :
                    break
                else :
                    moves = []

        train_board, train_pi, train_reward = self.get_train_data(moves)

        for i in range(len(moves)) :
            from tensorflow.keras.losses import CategoricalCrossentropy

            print("================ \n Search Board: \n", moves[i])
            print("\n Neural Network Input: \n", train_board[i])

            predictions = self.neural_network.predict(x = np.array([train_board[i]]))
            reward = predictions[0][0][0]
            pi = predictions[1][0]

            print("\nPredicted reward: ", reward, '\nPredicted probabilities: ', pi)
            #Print error
            print("Predicted policy error: ", CategoricalCrossentropy()(train_pi[i], pi).numpy())


    def get_boards(self, alphaZero):
        search_board = np.array([[0, 0, -1],
                                 [0, 1, 0],
                                 [0, 0, 0]])
        search_turn = 1

        buffer = alphaZero._buffer
        buffer_board = []

        for elem in buffer :
            board = elem[0][0]
            turn = elem[0][1]
            if (board == search_board).all() and search_turn == turn :
                buffer_board.append(elem)

        print(buffer_board)

    def print_buffer(self, alphaZero):
        buffer = alphaZero._buffer
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

    def test_predict(self, board, turn):
        x = [self.adapter.to_input(board, turn) for i in range(128)]

        start = time()
        
        print(start - time())

        start = time()

        print(start - time())