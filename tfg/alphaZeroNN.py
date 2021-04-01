import sys
sys.path.insert(0, '/Documents/Juan Carlos/Estudios/Universidad/5º Carrera/TFG Informatica/ImplementacionTFG')


class NeuralNetworkAZ:

    def __init__(self,
                 learning_rate,
                 regularizer_constant,
                 momentum,
                 input_dim,
                 output_dim,
                 residual_layers,
                 filters,
                 kernel_size):
        self.learning_rate = learning_rate
        self.regularizer_constant = regularizer_constant
        self.momentum = momentum
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.residual_layers = residual_layers
        
        self.model = self._create_model(filters, kernel_size)

    def _create_model(self, filters, kernel_size):
        import tensorflow as tf
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input
        from tensorflow.keras.optimizers import Adam

        input = Input(shape=self.input_dim)
        nn = self._create_convolutional_layer(input, filters, kernel_size)

        for i in range(self.residual_layers):
            nn = self._create_residual_layer(nn, filters, kernel_size)

        value_head = self._create_value_head(nn)
        
        policy_head = self._create_policy_head(nn)

        model = Model(inputs=[input], outputs=[value_head, policy_head])
        model.compile(#optimizer=SGD(learning_rate=self.learning_rate,
                      #              momentum=self.momentum),
                      optimizer=Adam(learning_rate=self.learning_rate),
                      loss={
                          'value_head': tf.keras.losses.MeanSquaredError(),
                          'policy_head': tf.keras.losses.CategoricalCrossentropy()
                      },
                      loss_weights={'value_head': 0.5, 'policy_head': 0.5}
                      )

        return model

    def _create_convolutional_layer(self, input, filters, kernel_size):
        # TODO podemos ponerles nombres a los layers para que se visualicen
        #  mejor
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
        
    def _create_value_head(self, input):
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
        # We are using 20 but they used 256 in the paper
        layer = Dense(
            20,
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

    def _create_policy_head(self, input):
        from tensorflow.keras.layers import (
            Conv2D, BatchNormalization, LeakyReLU, Dense, Flatten, Activation
        )
        from tensorflow.keras import regularizers

        layer = Conv2D(
            filters=2,
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
        
        layer = Activation('softmax', name = 'policy_head')(layer)
        
        #layer = tf.keras.backend.print_tensor(layer, message="PH Output: ")
        #layer = Lambda(lambda x : x, name = 'policy_head')(layer)
        
        return layer

    def _softmax_cross_entropy(self, y_true, y_predicted):
        import tensorflow as tf

        pi = y_true
        p = y_predicted
      
        #zero = tf.zeros(shape = tf.shape(pi), dtype=tf.float32)
        #where = tf.equal(pi, zero)
        #negatives = tf.fill(tf.shape(pi), -100.0) 
        
        #p = tf.where(where, negatives, p) #We are basically masking out illegal moves
        #tf.keras.backend.print_tensor(p, message="P: ")
        #tf.keras.backend.print_tensor(pi, message="Pi: ")
        
        loss = tf.keras.losses.CategoricalCrossentropy()(pi, p)

        return loss
    
    # def fit(self, x, y, batch_size, epochs, verbose, validation_split):
    def fit(self, *args, **kwargs):
        # return self.model.fit(x=x, y=y, batch_size=batch_size, epochs=epochs,
        #                       verbose=verbose,
        #                       validation_split=validation_split)
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
        plot_model(self.model, to_file=path, show_shapes = True)

    def summary_model(self):
        self.model.summary()
