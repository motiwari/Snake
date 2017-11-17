STEP_SIZE = 20
APPLE_IMAGE = "./block.png"
INITIAL_APPLE_VALUE = 100
STEPPING = False
DISCOUNT_FACTOR = 0.98

WINDOW_WIDTH = 80
WINDOW_HEIGHT = 80

WIDTH_TILES = int(WINDOW_WIDTH/STEP_SIZE)
HEIGHT_TILES = int(WINDOW_HEIGHT/STEP_SIZE)

SPEED = 50.0
# Directions (enum)
RIGHT = 0
LEFT = 1
UP = 2
DOWN = 3


# For tensorflow parameters
save_steps = 1  # save the model every 1,000 training steps
copy_steps = 1  # copy online DQN to target DQN every 10,000 training steps
discount_rate = 0.99
batch_size = 20
checkpoint_path = "./my_dqn.ckpt"

n_hidden = 20
input_width = int(WIDTH_TILES * HEIGHT_TILES * 4)
hidden_activation = None
n_outputs = 4  # 4 discrete actions are available
learning_rate = 0.001
momentum = 0.9


# n_steps = 4000000  # total number of training steps
# training_start = 10000  # start training after 10,000 game iterations
# training_interval = 4  # run a training step every 4 game iterations
# skip_start = 90  # Skip the start of every game (it's just waiting time).
# iteration = 0  # game iterations
# done = True # env needs to be reset
