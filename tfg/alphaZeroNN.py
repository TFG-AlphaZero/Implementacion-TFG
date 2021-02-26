import sys
sys.path.insert(0, '/Documents/Juan Carlos/Estudios/Universidad/5º Carrera/TFG Informatica/ImplementacionTFG')

import tensorflow as tf

from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, BatchNormalization, LeakyReLU, Dense, Flatten, add
from keras.optimizers import SGD
from keras import regularizers

class NeuralNetwork():

    def __init__(self, learning_rate, regularizer_constant, momentum, input_dim, output_dim, residual_layers, filters, kernel_size):
        self.learning_rate = learning_rate
        self.regularizer_constant = regularizer_constant
        self.momentum = momentum
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.residual_layers = residual_layers
        
        self.model = self._create_model(filters, kernel_size)

    def _create_model(self, filters, kernel_size):
        input = Input(shape = self.input_dim) #Parametros
        nn = self._create_convolutional_layer(input, filters, kernel_size)

        for i in range(self.residual_layers) :
            nn = self._create_residual_layer(nn, filters, kernel_size)

        value_head = self._create_value_head(nn)
        policy_head = self._create_policy_head(nn)

        model = Model(inputs=[input], outputs=[value_head, policy_head])
        model.compile(optimizer = SGD(learning_rate = self.learning_rate, momentum = self.momentum), 
                      loss = {'value_head' : 'mean_squared_error', 'policy_head' : self._softmax_cross_entropy},
                      loss_weights={'value_head' : 0.5, 'policy_head' : 0.5}) #Cambiar parametros
        
        return model

    def _create_convolutional_layer(self, input, filters, kernel_size):
        layer = Conv2D(filters = filters, kernel_size = kernel_size, data_format="channels_first", padding='same',use_bias=False, 
                       activation='linear', kernel_regularizer = regularizers.l2(self.regularizer_constant))(input)
        layer = BatchNormalization(axis=1)(layer)
        layer = LeakyReLU()(layer)

        return layer

    def _create_residual_layer(self, input, filters, kernel_size):
        layer = self._create_convolutional_layer(input, filters, kernel_size)

        layer = Conv2D(filters = filters, kernel_size = kernel_size, data_format="channels_first", padding='same',use_bias=False, 
                activation='linear', kernel_regularizer = regularizers.l2(self.regularizer_constant))(input)
        layer = BatchNormalization(axis=1)(layer)

        layer = add([input, layer]) #Skip connection

        layer = LeakyReLU()(layer)

        return layer
        
    def _create_value_head(self, input):
        layer = Conv2D(filters = 1, kernel_size = (1, 1), data_format="channels_first", padding='same', use_bias=False, 
                activation='linear', kernel_regularizer = regularizers.l2(self.regularizer_constant))(input)
        layer = BatchNormalization(axis=1)(layer)
        layer = LeakyReLU()(layer)

        layer = Flatten()(layer)
        layer = Dense(20, activation='linear', use_bias=False, kernel_regularizer = regularizers.l2(self.regularizer_constant))(layer) #Hemos puesto 20 pero en el paper son 256
        layer = LeakyReLU()(layer)
        
        layer = Dense(1, activation='tanh', use_bias=False, kernel_regularizer = regularizers.l2(self.regularizer_constant))(layer)

        return layer

    def _create_policy_head(self, input):
        layer = Conv2D(filters = 2, kernel_size = (1, 1), data_format="channels_first", padding='same', use_bias=False, 
                activation='linear', kernel_regularizer = regularizers.l2(self.regularizer_constant))(input)
        layer = BatchNormalization(axis=1)(layer)
        layer = LeakyReLU()(layer)

        layer = Flatten()(layer)
        layer = Dense(self.output_dim, activation='linear', use_bias=False, kernel_regularizer = regularizers.l2(self.regularizer_constant))(layer) #More parameters
        
        return layer

    def _softmax_cross_entropy(self, Y, Y_predicted):
        pi = Y
        
        zero = tf.zeros(shape = tf.shape(pi), dtype = tf.float32)
        where = tf.equal(pi, zero)
        negatives = tf.fill(tf.shape(pi), -100.0)
        prob = tf.where(where, negatives, Y_predicted)
        
        return tf.nn.softmax_cross_entropy_with_logits(labels = pi, logits = prob)
    
    def predict(self, X):
        return self.model.predict(X)

    def fit(self, X, Y, batch_size, epochs, verbose, validation_split):
        return self.model.fit(x=X, y=Y, batch_size = batch_size, epochs = epochs, verbose = verbose, validation_split = validation_split)
