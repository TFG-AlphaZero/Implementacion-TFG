import sys
sys.path.insert(0, '/Documents/Juan Carlos/Estudios/Universidad/5º Carrera/TFG Informatica/ImplementacionTFG')

import tensorflow as tf
import numpy as np
import tfg.alphaZeroConfig as config

from keras.models import Sequential, Model, load_model
from keras.layers import Input, Conv2D, BatchNormalization, LeakyReLU, Dense, Flatten, add, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras import regularizers


class NeuralNetworkAZ():

    def __init__(self, learning_rate, regularizer_constant, momentum, input_dim, output_dim, residual_layers, filters, kernel_size):
        self.learning_rate = learning_rate
        self.regularizer_constant = regularizer_constant
        self.momentum = momentum
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.residual_layers = residual_layers
        
        self.model = self._create_model(filters, kernel_size)

    def _create_model(self, filters, kernel_size):
        input = Input(shape = self.input_dim)
        nn = self._create_convolutional_layer(input, filters, kernel_size)

        for i in range(self.residual_layers) :
            nn = self._create_residual_layer(nn, filters, kernel_size)

        value_head = self._create_value_head(nn)
        
        policy_head = self._create_policy_head(nn)

        model = Model(inputs=[input], outputs=[value_head, policy_head])
        model.compile(optimizer = SGD(learning_rate = self.learning_rate, momentum = self.momentum), 
                      loss = {'value_head' : 'mean_squared_error', 'policy_head' : 'categorical_crossentropy'}, #Modified! (deberiamos llamar a nuestra funcion self._softmax_cross_entropy -> peta)
                      loss_weights={'value_head' : 0.5, 'policy_head' : 0.5})

        return model

    def _create_convolutional_layer(self, input, filters, kernel_size):
        layer = Conv2D(filters = filters, kernel_size = kernel_size, data_format="channels_last", padding='same',use_bias=False, 
                       activation='linear', kernel_regularizer = regularizers.l2(self.regularizer_constant))(input)
        layer = BatchNormalization(axis = -1)(layer) #Ojo! Usar axis = 1 si cambiamos a channels_first
        layer = LeakyReLU()(layer)

        return layer

    def _create_residual_layer(self, input, filters, kernel_size):
        layer = self._create_convolutional_layer(input, filters, kernel_size)

        layer = Conv2D(filters = filters, kernel_size = kernel_size, data_format="channels_last", padding='same',use_bias=False, 
                activation='linear', kernel_regularizer = regularizers.l2(self.regularizer_constant))(layer)
        layer = BatchNormalization(axis=-1)(layer)

        layer = add([input, layer]) #Skip connection

        layer = LeakyReLU()(layer)

        return layer
        
    def _create_value_head(self, input):
        layer = Conv2D(filters = 1, kernel_size = (1, 1), data_format="channels_last", padding='same', use_bias=False, 
                activation='linear', kernel_regularizer = regularizers.l2(self.regularizer_constant))(input)
        layer = BatchNormalization(axis=-1)(layer)
        layer = LeakyReLU()(layer)

        layer = Flatten()(layer)
        layer = Dense(20, activation='linear', use_bias=False, kernel_regularizer = regularizers.l2(self.regularizer_constant))(layer) #Hemos puesto 20 pero en el paper son 256
        layer = LeakyReLU()(layer)
        
        layer = Dense(1, activation='tanh', use_bias=False, kernel_regularizer = regularizers.l2(self.regularizer_constant), name = 'value_head')(layer)

        return layer

    def _create_policy_head(self, input):
        layer = Conv2D(filters = 2, kernel_size = (1, 1), data_format="channels_last", padding='same', use_bias=False, 
                activation='linear', kernel_regularizer = regularizers.l2(self.regularizer_constant))(input)
        layer = BatchNormalization(axis=-1)(layer)
        layer = LeakyReLU()(layer)

        layer = Flatten()(layer)
        #layer = Dense(self.output_dim, activation='linear', use_bias=False, kernel_regularizer = regularizers.l2(self.regularizer_constant), name = 'policy_head')(layer)
        
        layer = Dense(self.output_dim, activation='linear', use_bias=False, kernel_regularizer = regularizers.l2(self.regularizer_constant))(layer) #Modified!
        layer = Activation('softmax', name = 'policy_head')(layer) #Modified!

        return layer

    def _softmax_cross_entropy(self, Y, Y_predicted): #Not working properly
        pi = Y
        prob = Y_predicted
        
        zero = tf.zeros(shape = tf.shape(pi), dtype = tf.float32)
        where = tf.equal(pi, zero)
        negatives = tf.fill(tf.shape(pi), -100.0)
        prob = tf.where(where, negatives, prob)
        
        return tf.nn.softmax_cross_entropy_with_logits(labels = pi, logits = prob)
    
    def fit(self, X, Y, batch_size, epochs, verbose, validation_split):
        return self.model.fit(x=X, y=Y, batch_size = batch_size, epochs = epochs, verbose = verbose, validation_split = validation_split)

    def predict(self, X):
        return self.model.predict(X)
    
    def save_model(self, path):
        self.model.save(path)

    def load_model(self, path):
        self.model = load_model(path)

"""
#Example of use provided below!

input_dimension = (3, 3) + config.INPUT_LAYERS
output_dimension = 9

nn_test = NeuralNetworkAZ(learning_rate= config.LEARNING_RATE, regularizer_constant = config.REGULARIZER_CONST, momentum = config.MOMENTUM,
                          input_dim = input_dimension, output_dim = output_dimension, 
                          residual_layers=config.RESIDUAL_LAYERS, filters = config.CONV_FILTERS, kernel_size=config.CONV_KERNEL_SIZE)
#nn_test.model.summary()

first_player = np.array([[0,0,0], 
                         [0,1,0], 
                         [0,0,0]])

second_player = np.array([[0,0,0], 
                          [0,0,0], 
                          [0,0,1]])
sample_1 = np.reshape(np.append(first_player, second_player), input_dimension)

train_X = np.array([sample_1])
train_Y = np.array([-1])
train_Z = np.array([[1,0,0,0,0,0,0,0,0]])

nn_test.fit(X = train_X, Y = [train_Y, train_Z], batch_size = 1, epochs = 10, verbose = 2, validation_split = 0)
predictions = nn_test.predict(X = train_X)
print(predictions)
"""