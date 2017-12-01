import tensorflow as tf

STEP_SIZE = 20
APPLE_IMAGE = "./block.png"
INITIAL_APPLE_VALUE = 100
STEPPING = False

# not sure this is actually used anywhere
DISCOUNT_FACTOR = 0.98

WINDOW_WIDTH = 280
WINDOW_HEIGHT = 280

WIDTH_TILES = int(WINDOW_WIDTH/STEP_SIZE)
HEIGHT_TILES = int(WINDOW_HEIGHT/STEP_SIZE)

SPEED = 100
# Directions (enum)
RIGHT = 0
LEFT = 1
UP = 2
DOWN = 3

# For tensorflow parameters
save_steps = 50  # save the model every 1,000 training steps
copy_steps = 300  # copy online DQN to target DQN every 10,000 training steps
discount_rate = 0.98
batch_size = 100
training_start = 1000  # start training after game's memory has built up to 'training_start'
num_updates_per_game = 20
checkpoint_path = "./my_dqn.ckpt"
replay_memory = 100000

# convolutional parameters
conv_n_maps = [32, 64]
conv_kernel_sizes = [(4,4), (4,4)]
conv_strides = [2, 2]
conv_paddings = ["SAME"] * 2
conv_activation = [tf.nn.relu] * 2

n_hidden1 = 300
input_height = int(HEIGHT_TILES) + 2
input_width = int(WIDTH_TILES) + 2
input_channels = 5 # we have a matrix for head, tail, apple, board, and body all stacked on top of each other
hidden_activation = None
n_hidden_in = 1024
n_outputs = 4  # 4 discrete actions are available
learning_rate = 0.001
momentum = 0.95

eps_min = 0.01
eps_max = .5
eps_decay_steps = 2000

# n_steps = 4000000  # total number of training steps

# training_interval = 4  # run a training step every 4 game iterations
# skip_start = 90  # Skip the start of every game (it's just waiting time).
# iteration = 0  # game iterations
# done = True # env needs to be reset
