import tensorflow as tf

STEP_SIZE = 20
APPLE_IMAGE = "./block.png"
INITIAL_APPLE_VALUE = 100
STEPPING = False

# Used to decay the apple
DISCOUNT_FACTOR = 1.00 # i turned off decay for now to help the snake train

WINDOW_WIDTH = 200
WINDOW_HEIGHT = 200

WIDTH_TILES = int(WINDOW_WIDTH/STEP_SIZE)
HEIGHT_TILES = int(WINDOW_HEIGHT/STEP_SIZE)

SPEED = 100
# Directions (enum)
RIGHT = 0
LEFT = 1
UP = 2
DOWN = 3

# For tensorflow parameters
save_steps = 1000  # save the model every 1,000 training steps
copy_steps = 10000  # copy online DQN to target DQN every 'copy_steps' training steps
discount_rate = 0.95 # discount rate for the q-value algorithm
batch_size = 50
training_start = 10000  # start training after game's memory has built up to 'training_start'
num_updates_per_game = 5 #this is not currently being used
checkpoint_path = "./my_dqn.ckpt"
replay_memory = 500000
epsilon_guided = .65 # this controls, when an exploration action is chosen, how often it chooses a guided action

# convolutional parameters
conv_n_maps = [32, 64, 64]
conv_kernel_sizes = [(4,4), (4,4), (4,4)]
conv_strides = [1, 2, 1]
conv_paddings = ["SAME"] * 3
conv_activation = [tf.nn.elu] * 3

n_hidden1 = 200
input_height = int(HEIGHT_TILES) + 2
input_width = int(WIDTH_TILES) + 2
input_channels = 6 # we have a matrix for head, tail, apple, board, secnd body part, and rest of body all stacked on top of each other
hidden_activation = tf.nn.elu
n_hidden_in = 2304
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
