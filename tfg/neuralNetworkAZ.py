import sys
sys.path.insert(0, '/Documents/Juan Carlos/Estudios/Universidad/5º Carrera/TFG Informatica/ImplementacionTFG')

from keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, LeakyReLU, Dense, Flatten 
from tensorflow.keras.optimizers import SGD


class NeuralNetworkAZ():

    def __init__(self, learning_rate = 0.01, input_dim = 1, output_dim = 1, hidden_layers = 40):
        self.learning_rate = learning_rate
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.model = self._create_model()

    def _create_model(self, filters = 256, kernel_size = (3, 3)):
        input = Input() #Parametros
        nn = self._create_convolutional_layer(input, filters, kernel_size)

        for i in range(self.hidden_layers) :
            nn = self._create_residual_layer(nn, filters, kernel_size)

        value_head = self._create_value_head(nn)
        policy_head = self._create_policy_head(nn)

        model = Model(inputs=[input], outputs=[value_head, policy_head])
        model.compile(optimizer = SGD(learning_rate = self.learning_rate), 
                      loss = {'value_head' : 'mean_squared_error', 'policy_head' : 'mean_squared_error'}) #Cambiar parametros
        
        return model

    def _create_residual_layer(self, input, filters, kernel_size):
        layer = self._create_convolutional_layer(input, filters, kernel_size)
        
        layer = Conv2D(filters = filters, kernel_size=kernel_size, activation='linear') #More parameters
        layer = BatchNormalization(axis=1)(layer)

        layer = add([input, layer]) #Skip connection

        layer = LeakyReLU()(layer)

        return layer
    
    def _create_convolutional_layer(self, input, filters, kernel_size):
        layer = Conv2D(filters = filters, kernel_size=kernel_size, activation='linear') #More parameters
        layer = BatchNormalization(axis=1)(layer)
        layer = LeakyReLU()(layer)

        return layer
    
    def _create_value_head(self):
        layer = Conv2D(filters = 1, kernel_size=(1, 1), activation='linear')
        layer = BatchNormalization(axis=1)(layer)
        layer = LeakyReLU()(layer)

        layer = Flatten()(layer)
        layer = Dense(20, activation='linear', use_bias=False)(layer) #Faltan parametros
        layer = LeakyReLU()(layer)
        
        layer = Dense(1, activation='tanh', use_bias=False)(layer)

        return layer

    def _create_policy_head(self):
        layer = Conv2D(filters = 2, kernel_size = (1, 1), activation = 'linear')
        layer = BatchNormalization(axis=1)(layer)
        layer = LeakyReLU()(layer)

        layer = Flatten()(layer)
        layer = Dense(self.output_dim, activation='linear', use_bias=False)(layer) #More parameters
        
        return layer

    def predict(self, data):
        return self.model.predict(data)

    def fit(self):
        return self.model.fit()



    
