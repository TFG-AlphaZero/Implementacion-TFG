import sys
sys.path.insert(0, '/Documents/Juan Carlos/Estudios/Universidad/5º Carrera/TFG Informatica/ImplementacionTFG')


class NeuralNetworkAZ:
    """Class implementing AlphaZero's neural network."""

    def __init__(self,
                 learning_rate,
                 regularizer_constant,
                 momentum,
                 input_dim,
                 output_dim,
                 residual_layers,
                 filters,
                 kernel_size):
        """

        Args:
            learning_rate (float): Network's learning rate.
            regularizer_constant (float): Constant used in regularization.
            momentum (float): Unused at the moment.
            input_dim ((int, int, int)): Expected input tensors' shape.
            output_dim (int): Number of outputs.
            residual_layers (int): Number of residual layers.
            filters (int): Number of filters per convolution.
            kernel_size ((int, int) or int): Kernel size of every convolution.

        """
        self.learning_rate = learning_rate
        self.regularizer_constant = regularizer_constant
        self.momentum = momentum
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.residual_layers = residual_layers
        
        self.model = self._create_model(filters, kernel_size)

    def _create_model(self, filters, kernel_size):
        from tensorflow.keras.losses import (
            MeanSquaredError, CategoricalCrossentropy
        )
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input
        from tensorflow.keras.optimizers import SGD

        input = Input(shape=self.input_dim)
        nn = self._create_convolutional_layer(input, filters, kernel_size)

        for i in range(self.residual_layers):
            nn = self._create_residual_layer(nn, filters, kernel_size)

        value_head = self._create_value_head(nn, filters)
        
        policy_head = self._create_policy_head(nn, filters)

        model = Model(inputs=[input], outputs=[value_head, policy_head])
        model.compile(optimizer=SGD(learning_rate=self.learning_rate),
                      loss={
                          'value_head': MeanSquaredError(),
                          'policy_head': CategoricalCrossentropy()
                      },
                      loss_weights={'value_head': 0.5, 'policy_head': 0.5}
                      )

        return model

    def _create_convolutional_layer(self, input, filters, kernel_size):
        from tensorflow.keras.layers import (
            Conv2D, BatchNormalization, LeakyReLU
        )
        from tensorflow.keras import regularizers

        layer = Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            data_format="channels_last",
            padding='same',
            use_bias=False,
            activation='linear',
            kernel_regularizer=regularizers.l2(self.regularizer_constant)
        )(input)
        # Use axis=1 if channels_first
        layer = BatchNormalization(axis=-1)(layer)
        layer = LeakyReLU()(layer)

        return layer

    def _create_residual_layer(self, input, filters, kernel_size):
        from tensorflow.keras.layers import (
            Conv2D, BatchNormalization, LeakyReLU, add
        )
        from tensorflow.keras import regularizers

        layer = self._create_convolutional_layer(input, filters, kernel_size)

        layer = Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            data_format="channels_last",
            padding='same',
            use_bias=False,
            activation='linear',
            kernel_regularizer=regularizers.l2(self.regularizer_constant)
        )(layer)
        layer = BatchNormalization(axis=-1)(layer)

        # Skip connection
        layer = add([input, layer])

        layer = LeakyReLU()(layer)

        return layer
        
    def _create_value_head(self, input, size):
        from tensorflow.keras.layers import (
            Conv2D, BatchNormalization, LeakyReLU, Dense, Flatten
        )
        from tensorflow.keras import regularizers

        layer = Conv2D(
            filters=1,
            kernel_size=(1, 1),
            data_format="channels_last",
            padding='same',
            use_bias=False,
            activation='linear',
            kernel_regularizer=regularizers.l2(self.regularizer_constant)
        )(input)
        layer = BatchNormalization(axis=-1)(layer)
        layer = LeakyReLU()(layer)

        layer = Flatten()(layer)
        # They used 256 in the paper
        layer = Dense(
            size,
            activation='linear',
            use_bias=False,
            kernel_regularizer=regularizers.l2(self.regularizer_constant)
        )(layer)
        layer = LeakyReLU()(layer)
        
        layer = Dense(
            1,
            activation='tanh',
            use_bias=False,
            kernel_regularizer=regularizers.l2(self.regularizer_constant),
            name='value_head'
        )(layer)

        return layer

    def _create_policy_head(self, input, filters):
        from tensorflow.keras.layers import (
            Conv2D, BatchNormalization, LeakyReLU, Dense, Flatten, Activation
        )
        from tensorflow.keras import regularizers

        layer = Conv2D(
            filters=filters,
            kernel_size=(1, 1),
            data_format="channels_last",
            padding='same',
            use_bias=False,
            activation='linear',
            kernel_regularizer=regularizers.l2(self.regularizer_constant)
        )(input)
        layer = BatchNormalization(axis=-1)(layer)
        layer = LeakyReLU()(layer)

        layer = Flatten()(layer)

        layer = Dense(
            self.output_dim,
            activation='linear',
            use_bias=False,
            kernel_regularizer=regularizers.l2(self.regularizer_constant)
        )(layer)
        
        layer = Activation('softmax', name='policy_head')(layer)
        
        return layer

    def fit(self, *args, **kwargs):
        return self.model.fit(*args, **kwargs)

    def predict(self, x):
        # Much faster than predict for small inputs
        predictions = self.model(x, training=False)
        # Convert tensors to numpy arrays
        res = predictions[0].numpy(), predictions[1].numpy()
        # Output as a tuple (reward, probabilities)
        return res

    def save_model(self, path):
        self.model.save(path)

    def load_model(self, path):
        from tensorflow.keras.models import load_model
        self.model = load_model(path)

    def plot_model(self, path):
        from tensorflow.keras.utils import plot_model
        plot_model(self.model, to_file=path, show_shapes=True)

    def summary_model(self):
        self.model.summary()
