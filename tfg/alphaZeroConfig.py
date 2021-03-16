"""In this file, every alphaZero parameter is set"""

# 1) Self Play
C_PUCT = 5
MCTS_TIMES = 40
SELF_PLAY_TIMES = 1
T_EQUALS_ONE = 12
BUFFER_SIZE = 4096
MAX_WORKERS = None

# 2) Retrain Network
MAX_TRAIN_TIME = 600
MAX_TRAIN_ERROR = 0.01
LEARNING_RATE = 0.001
BATCH_SIZE = 128
EPOCHS = 25
REGULARIZER_CONST = 0.0001
MOMENTUM = 0.9
RESIDUAL_LAYERS = 7
CONV_FILTERS = 256
CONV_KERNEL_SIZE = (3, 3)
INPUT_LAYERS = (2,)
