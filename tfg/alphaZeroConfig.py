"""In this file, every alphaZero parameter is set"""

# 1) MCTS Parameters
MCTS_ITER = 50  # At least two iterations required
C_PUCT = 1
MCTS_MAX_TIME = None
EXPLORATION_NOISE = (0.25, 1)  # (noise_fraction, Dirichlet_alpha)

# 2) Neural Network Parameters
LEARNING_RATE = 0.01
EPOCHS = 10
REGULARIZER_CONST = 0.0001
MOMENTUM = 0.9
RESIDUAL_LAYERS = 19
CONV_FILTERS = 64
CONV_KERNEL_SIZE = (3, 3)

# 3) Train Parameters
MAX_TRAIN_TIME = None
MIN_TRAIN_ERROR = None
MAX_GAMES_COUNTER = 100

SELF_PLAY_TIMES = 10
TEMPERATURE = 12
BUFFER_SIZE = 4096
BATCH_SIZE = 128
MAX_WORKERS = None


class AlphaZeroConfig:
    """Wrapper for Neural Network Parameters."""

    def __init__(self,
                 learning_rate=LEARNING_RATE,
                 regularizer_constant=REGULARIZER_CONST,
                 momentum=MOMENTUM,
                 residual_layers=RESIDUAL_LAYERS,
                 filters=CONV_FILTERS,
                 kernel_size=CONV_KERNEL_SIZE):
        self.learning_rate = learning_rate
        self.regularizer_constant = regularizer_constant
        self.momentum = momentum
        self.residual_layers = residual_layers
        self.filters = filters
        self.kernel_size = kernel_size
