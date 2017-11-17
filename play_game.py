from random import randint
import pygame
import time
import apple
import snake
import state
import config
import gameengine
import sys
import random
import argparse
import pickle
import tensorflow as tf
import numpy as np
from collections import deque
import os

gwidth = config.DEFAULT_WINDOW_WIDTH/config.STEP_SIZE
gheight = config.DEFAULT_WINDOW_HEIGHT/config.STEP_SIZE
n_hidden = 20
input_width = int(gwidth * gheight * 2)
hidden_activation = None
n_outputs = 4  # 4 discrete actions are available

learning_rate = 0.001
momentum = 0.95
replay_memory_size = 500000
replay_memory = deque([], maxlen=replay_memory_size)




#convert game state into a feature vector
def preprocess_observation(obs):
    width = int(config.DEFAULT_WINDOW_WIDTH/config.STEP_SIZE)
    height = int(config.DEFAULT_WINDOW_HEIGHT/config.STEP_SIZE)

    #add indicators at each coordinate for apple
    a = [0] * width * height
    xa = int(obs.apple[0])
    ya = int(obs.apple[1])
    a[width * ya + xa] = 1

    #add indicators at each coordinate for snake
    b =  [0] * width * height

    x = obs.head[0]
    y = obs.head[1]
    #if statement to account for when head goes off board
    if x >= 0 and y >= 0 and x <width and y < height:
        b[int(width * y + x)] = 1

    x = obs.tail[0]
    y = obs.tail[1]
    #if statement to account for when head goes off board
    if x >= 0 and y >= 0 and x <width and y < height:
        b[int(width * y + x)] = 1

    for w in obs.body_parts:
        x = w[0]
        y = w[1]
        if x >= 0 and y >= 0 and x <width and y < height:
            b[int(width * y + x)] = 1

    return np.array(a + b)

def q_network(X_state, name):
    prev_layer = X_state
    with tf.variable_scope(name) as scope:
        hidden = tf.layers.dense(prev_layer, n_hidden,
                                 activation=hidden_activation,
                                 kernel_initializer=initializer)
        outputs = tf.layers.dense(hidden, n_outputs,
                                  kernel_initializer=initializer)
    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       scope=scope.name)
    trainable_vars_by_name = {var.name[len(scope.name):]: var
                              for var in trainable_vars}
    return outputs, trainable_vars_by_name

hidden_activation = tf.nn.relu
initializer = tf.contrib.layers.variance_scaling_initializer()
X_state = tf.placeholder(tf.float32, shape=[None, input_width])
online_q_values, online_vars = q_network(X_state, name="q_networks/online")
target_q_values, target_vars = q_network(X_state, name="q_networks/target")

copy_ops = [target_var.assign(online_vars[var_name])
            for var_name, target_var in target_vars.items()]
copy_online_to_target = tf.group(*copy_ops)

X_action = tf.placeholder(tf.int32, shape=[None])
q_value = tf.reduce_sum(online_q_values * tf.one_hot(X_action, n_outputs),
                        axis=1, keep_dims=True)

ytrain = tf.placeholder(tf.float32, shape=[None, 1])
error = tf.abs(ytrain - q_value)
clipped_error = tf.clip_by_value(error, 0.0, 1.0)
linear_error = 2 * (error - clipped_error)
loss = tf.reduce_mean(tf.square(clipped_error) + linear_error)

global_step = tf.Variable(0, trainable=False, name='global_step')
optimizer = tf.train.MomentumOptimizer(learning_rate, momentum, use_nesterov=True)
training_op = optimizer.minimize(loss, global_step=global_step)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

from datetime import datetime

#The next few lines are there for the purpose of being able to view things on tensorboard
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

eps_min = 0.1
eps_max = 1.0
eps_decay_steps = 2000

def epsilon_greedy(q_values, step):
    #print(step)
    epsilon = eps_min
    #epsilon = max(eps_min, eps_max - (eps_max-eps_min) * step/eps_decay_steps)
    if np.random.rand() < epsilon:
        return np.random.randint(n_outputs) # random action
    else:
        print(q_values)
        return np.argmax(q_values) # optimal action

n_steps = 4000000  # total number of training steps
training_start = 10000  # start training after 10,000 game iterations
training_interval = 4  # run a training step every 4 game iterations
save_steps = 1  # save the model every 1,000 training steps
copy_steps = 1  # copy online DQN to target DQN every 10,000 training steps
discount_rate = 0.99
skip_start = 90  # Skip the start of every game (it's just waiting time).
batch_size = 50
iteration = 0  # game iterations
checkpoint_path = "./my_dqn.ckpt"
done = True # env needs to be reset

def isNextMoveCollision(pyg, direction):
    dummy_head = None
    x_change = {
        config.RIGHT: pyg.snake.step,
        config.LEFT: -pyg.snake.step,
        config.UP: 0,
        config.DOWN: 0,
    }
    y_change = {
        config.RIGHT: 0,
        config.LEFT: 0,
        config.UP: -pyg.snake.step,
        config.DOWN: pyg.snake.step,
    }
    dummy_head = snake.Head(pyg.snake.x[0] + x_change[direction], pyg.snake.y[0] + y_change[direction])

    # Check Board collision
    if dummy_head.x < 0 or dummy_head.x >= pyg.windowWidth or \
        dummy_head.y < 0 or dummy_head.y >= pyg.windowHeight:
        return True
    # Check Snake collision
    # Need to account for the fact that the snake will have moved by 1,
    # so we don't start on 3rd segment of snake
    for i in range(1, pyg.snake.length - 1):
        dummy_head2 = snake.Head(pyg.snake.x[i], pyg.snake.y[i])
        if pyg.gameEngine.isCollision(dummy_head, dummy_head2):
            return True
    return False

class App:
    windowWidth = config.DEFAULT_WINDOW_WIDTH
    windowHeight = config.DEFAULT_WINDOW_HEIGHT
    boardWidth = windowWidth/config.STEP_SIZE
    boardHeight = windowHeight/config.STEP_SIZE
    snake = None
    apple = None
    verbose = False

    def __init__(self, args):
        self._running = True
        self._display_surf = None
        self._image_surf = None
        self._apple_surf = None
        self.gameEngine = gameengine.GameEngine()
        self.snake = snake.Snake(3)
        self.apple = apple.Apple(5,5)
        self.apple.x, self.apple.y = random.choice(self.gameEngine.getBoardFreeSquares(self.snake))
        self.usingAI = args.ai
        self.usingNN = args.ain
        self.verbose = args.verbose
        self.saveHistory = args.history
        self.history = []
        self.actionHistory = []
        self.counter = 0

    def on_init(self):
        pygame.init()
        self._display_surf = pygame.display.set_mode((self.windowWidth,self.windowHeight), pygame.HWSURFACE)

        pygame.display.set_caption('SNAKE - motiwari, rschoenh, benzhou')
        self._running = True
        self._image_surf = pygame.image.load("pygame.png").convert()
        self._apple_surf = pygame.image.load("block.png").convert()

    def on_event(self, event):
        if event.type == pygame.QUIT:
            self._running = False

    def on_loop(self):
        self.snake.update()
        self.apple.update()
        # Does snake collide with itself?

        for i in range(2, self.snake.length):
            # Fix this
            dummy_head = snake.Head(self.snake.x[i], self.snake.y[i])
            if self.gameEngine.isCollision(self.snake.head, dummy_head):
                self.snake.addActionAndReward(self.snake.direction, 0)
                if self.verbose:
                    print("You lose! Collision: ")
                    print("FINAL SCORE: ", self.snake.score)
                    print(self.snake.ars)
                self.get_state()
                self._running = False
                #exit(0)

        if self.snake.head.x < 0 or self.snake.head.x >= self.windowWidth or \
            self.snake.head.y < 0 or self.snake.head.y >= self.windowHeight:
            self.snake.addActionAndReward(self.snake.direction, 0)
            if self.verbose:
                print("You lose! Off the board!")
                print("FINAL SCORE: ", self.snake.score)
                print(self.snake.ars)
            self.get_state()
            self._running = False
            #exit(0)

        # Does snake eat apple?
        if self.gameEngine.isCollision(self.apple, self.snake.head):
            self.snake.length = self.snake.length + 1
            self.snake.score += self.apple.value
            if self.verbose:
                print("Ate apple with value ", self.apple.value)
            self.snake.addActionAndReward(self.snake.direction, self.apple.value)
            self.apple.value = 100
            freeSqs = self.gameEngine.getBoardFreeSquares(self.snake)
            if freeSqs == []:
                if self.verbose:
                    print("You WON Snake!!")
                    print("FINAL SCORE: ", self.snake.score)
                self.get_state()
                exit(0)
            else:
                self.apple.x, self.apple.y = random.choice(freeSqs)
        else:
            self.snake.addActionAndReward(self.snake.direction, 0)

    def on_render(self):
        self._display_surf.fill((100,100,0))
        self.snake.draw(self._display_surf, self._image_surf)
        self.apple.draw(self._display_surf, self._apple_surf)
        pygame.display.flip()

    def on_cleanup(self):
        pygame.quit()

    def on_execute(self):
        if self.on_init() == False:
            self._running = False


        while( self._running ):
            pygame.event.pump()
            keys = pygame.key.get_pressed()

            if(self.usingAI):
                self.choose_ai_move()

            # Not using AI, Snake is player-controlled
            else:
                self.use_player_move(keys)

            if (keys[pygame.K_ESCAPE]):
                self._running = False

            self.on_loop()
            self.on_render()
            self.counter += 1
            # TODO: WTF is going on here?
            if self.counter % 3 ==0:
                self.get_state()

            #time.sleep((100.0 - config.SPEED) / 1000.0);

            #save game STATE
            if(self.saveHistory):
                if(self.history):
                    if self.history[-1] != state.State(self): #make sure that the state has changed before we append a new state
                        self.history.append(state.State(self))
                        self.actionHistory.append(self.snake.direction)
                else:
                    self.history.append(state.State(self))
                    self.actionHistory.append(self.snake.direction)


        #pickle.dump(self.history,open('gamehistory.pkl','wb'))
        self.on_cleanup()

        return self.history, self.actionHistory

    def use_player_move(self, keys):
        # Interpret keystroke
        if keys[pygame.K_LEFT] and self.snake.last_moved != config.RIGHT:
            self.snake.moveLeft()
        if keys[pygame.K_RIGHT] and self.snake.last_moved != config.LEFT:
            self.snake.moveRight()
        if keys[pygame.K_DOWN] and self.snake.last_moved != config.UP:
            self.snake.moveDown()
        if keys[pygame.K_UP] and self.snake.last_moved != config.DOWN:
            self.snake.moveUp()

    def choose_ai_move(self):
        x = self.snake.x[0]
        y = self.snake.y[0]
        d = self.snake.last_moved

        # Move closer to the apple
        if self.usingNN:
                # Online DQN evaluates what to do
                s = state.State(self)
                features = preprocess_observation(s)
                action = 0
                with tf.Session() as sess:
                    if os.path.isfile(checkpoint_path + ".index"):
                            saver.restore(sess, checkpoint_path)
                    else:
                        init.run()
                        copy_online_to_target.run()
                    step = global_step.eval()
                    q_values = online_q_values.eval(feed_dict={X_state: [features]})
                    #print(features)
                    #print(q_values)
                    action = epsilon_greedy(q_values, step)
                    #CHECK TO MAKE SURE THAT CHOSEN DIRECTION IS VALID
                if d == config.RIGHT:
                    if action != config.LEFT:
                        self.snake.direction = action
                    else:
                        self.snake.direction = d
                elif d == config.LEFT:
                    if action != config.RIGHT:
                        self.snake.direction = action
                    else:
                        self.snake.direction = d
                elif d == config.UP:
                    if action != config.DOWN:
                        self.snake.direction = action
                    else:
                        self.snake.direction = d
                elif d == config.DOWN:
                    if action != config.UP:
                        self.snake.direction = action
                    else:
                        self.snake.direction = d
        else:
            if self.apple.x - x < 0 and d != config.RIGHT and not isNextMoveCollision(self, config.LEFT):  # Make sure snake isn't moving right
                self.snake.moveLeft()
            elif self.apple.x - x > 0 and d != config.LEFT and not isNextMoveCollision(self, config.RIGHT): # Make sure snake isn't moving left
                self.snake.moveRight()
            elif self.apple.y - y < 0 and d != config.DOWN and not isNextMoveCollision(self, config.UP): # Make sure snake isn't moving down
                self.snake.moveUp()
            elif self.apple.y - y > 0 and d != config.UP and not isNextMoveCollision(self, config.DOWN): # Make sure snake isn't moving up
                self.snake.moveDown()
            else:
                # Case when apple is directly behind snake
                if (d == config.LEFT or d == config.RIGHT) and not isNextMoveCollision(self, d + 2):
                    self.snake.direction = d + 2
                elif (d == config.UP or d == config.DOWN) and not isNextMoveCollision(self, d - 2):
                    self.snake.direction = d - 2
                else:
                    x = list(range(0,4))
                    random.shuffle(x)
                    for i in x: # Iterate until you find a valid move
                        if not isNextMoveCollision(self, i):
                            self.snake.direction = i
                            break

                    # No move exists, move right. THIS IS WHERE YOU DIE.
                    self.snake.direction = config.RIGHT


    def get_state(self):
        s = state.State(self)
        if self.verbose:
            print("NEW STATE:")
            print("SCORE: ", s.score)
            print("APPLE: ", s.apple)
            print("HEAD: ", s.head)
            print("TAIL: ", s.tail)
            print("BODY PARTS: ", s.body_parts)


def get_args(arguments):
    parser = argparse.ArgumentParser(description=__doc__,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
    # For generating pairs of sites
    parser.add_argument('-a', '--ai', help='Use AI', action='store_true')
    parser.add_argument('-an', '--ain', help='Use AI NeuralNetwork', action='store_true') #if this flag is not set, it will default to baseline
    parser.add_argument('-v', '--verbose', help='Verbose output', action='store_true')
    parser.add_argument('-p', '--history', help='Collect and Save State History', action='store_true')
    args = parser.parse_args(arguments)
    return args

def update(gameHistory):
    with tf.Session() as sess:
        if os.path.isfile(checkpoint_path + ".index"):
            saver.restore(sess, checkpoint_path)
        else:
            init.run()
            copy_online_to_target.run()

        for X_state_val, X_action_val, rewards, X_next_state_val, continues in gameHistory:
            step = global_step.eval()
            if step >= n_steps:
                break



            #if iteration < training_start or iteration % training_interval != 0:
            #    continue

            # Sample memories and use the target DQN to produce the target Q-Value
            #X_state_val, X_action_val, rewards, X_next_state_val, continues = (
            #    sample_memories(batch_size))
            X_next_state_val = np.array(X_next_state_val).reshape(1,input_width)
            X_state_val = np.array(X_state_val).reshape(1,input_width)
            X_action_val = np.array(X_action_val).reshape(1)
            #print(X_state_val, X_action_val, 'rewards', rewards, X_next_state_val, 'continues', continues )
            next_q_values = target_q_values.eval(
                feed_dict={X_state: X_next_state_val})
            max_next_q_values = np.max(next_q_values, axis=1, keepdims=True)
            y_val = rewards + continues * discount_rate * max_next_q_values

            # Train the online DQN
            training_op.run(feed_dict={X_state: X_state_val,
                                       X_action: X_action_val, ytrain: y_val})

        #variable_check_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                           #scope="q_networks/online")
        #for var in variable_check_list:
        #    print(var)
        #    print(var.eval())
            # Regularly copy the online DQN to the target DQN
        if step % copy_steps == 0:
            copy_online_to_target.run()

            # And save regularly
        if step % save_steps == 0:
            saver.save(sess, checkpoint_path)

def pre_processHistory(stateHist,actionHist):
        h = []
        for i,s in enumerate(stateHist):
            if i > 0:
                reward = s.score - stateHist[i-1].score
                newState = preprocess_observation(s)
                oldState = preprocess_observation(stateHist[i-1])
                action = actionHist[i-1]
                #is this the last state of the episode? If yes, cont = 0 (cont = continue)
                cont = 1
                if i + 1 == len(stateHist):
                    cont = 0

                h.append((oldState,action,reward,newState,cont))

        return h

if __name__ == "__main__" :
    args = get_args(sys.argv[1:])

    theApp = App(args)
    stateHist,actionHist = theApp.on_execute()
    if args.history:
        gameHistory = pre_processHistory(stateHist,actionHist)
        update(gameHistory)
