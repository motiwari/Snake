import config as cnfg
import tensorflow as tf
import numpy as np
import os
from datetime import datetime

def sample_memories(replay_memory,batch_size):
    indices = np.random.permutation(len(replay_memory))[:batch_size]
    cols = [[], [], [], [], []] # state, action, reward, next_state, continue
    for idx in indices:
        memory = replay_memory[idx]
        for col, value in zip(cols, memory):
            col.append(value)
    cols = [np.array(col) for col in cols]
    return (cols[0], cols[1], cols[2].reshape(-1, 1), cols[3],
            cols[4].reshape(-1, 1))

def update(gameHistory, sess):
    # TODO: Reloading is very slow
    if os.path.isfile(cnfg.checkpoint_path + ".index"):
        saver.restore(sess, cnfg.checkpoint_path)
    else:
        init.run()
        copy_online_to_target.run()
    step = global_step.eval()
    for i in range(cnfg.num_updates_per_game):
        X_state_val, X_action_val, rewards, X_next_state_val, continues = sample_memories(gameHistory,cnfg.batch_size)

        #if len(gameHistory) < 100: #or iteration % training_interval != 0:
        #   return

        # Sample memories and use the target DQN to produce the target Q-Value
        #X_state_val, X_action_val, rewards, X_next_state_val, continues = (
        #    sample_memories(batch_size))
        #X_next_state_val = np.array(X_next_state_val).reshape(1,cnfg.input_width)
        #X_state_val = np.array(X_state_val).reshape(1,cnfg.input_width)
        #X_action_val = np.array(X_action_val).reshape(1)
        #print(X_state_val, X_action_val, 'rewards', rewards, X_next_state_val, 'continues', continues )
    #for i in range(50):

        next_q_values = target_q_values.eval(
            feed_dict={X_state: X_next_state_val})
        max_next_q_values = np.max(next_q_values, axis=1, keepdims=True)
        y_val = rewards + continues * cnfg.discount_rate * max_next_q_values

        # if i > 19000:
        #     print('error', error.eval(feed_dict={X_state: X_state_val,
        #                             X_action: X_action_val, ytrain: y_val}))

    # print('q_value', q_value.eval(feed_dict={X_state: X_state_val,
    #                            X_action: X_action_val, ytrain: y_val}))
    # print('ytrain', ytrain.eval(feed_dict={X_state: X_state_val,
    #                            X_action: X_action_val, ytrain: y_val}))
    # for a,b,c,d in zip(y_val,X_state_val,X_action_val,continues):
        #    print(a)
        #    print(b)
        #    print(c)
        #    print(d)
        #    print('\n')
        # Train the online DQN
        #for i in range(300):
        #q_values = online_q_values.eval(feed_dict={X_state: X_state_val})
            #print(q_values)
        training_op.run(feed_dict={X_state: X_state_val,
                                   X_action: X_action_val, ytrain: y_val})

    #variable_check_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       #scope="q_networks/online")
    #for var in variable_check_list:
    #    print(var)
    #    print(var.eval())
    # Regularly copy the online DQN to the target DQN
    # if step % cnfg.copy_steps == 0:
    #     copy_online_to_target.run()
    #
    # # And save regularly
    # if step % cnfg.save_steps == 0:
    #     saver.save(sess, cnfg.checkpoint_path)

def pre_processHistory(stateHist, actionHist):
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

                h.append((oldState, action, reward, newState, cont))

        return h


#convert game state into a feature vector
def preprocess_observation(obs):
    # Add indicators at each coordinate for apple
    OFFSET = 1
    PAD = 2
    a = np.zeros((cnfg.HEIGHT_TILES + PAD,cnfg.WIDTH_TILES + PAD))
    apple_x = int(obs.apple[0])
    apple_y = int(obs.apple[1])
    a[apple_y + OFFSET, apple_x + OFFSET] = 1

    # Add indicators for head
    h = np.zeros((cnfg.HEIGHT_TILES + PAD,cnfg.WIDTH_TILES + PAD))

    head_x = obs.head[0]
    head_y = obs.head[1]
    # If statement to account for when head goes off board
    if head_x >= 0 and head_y >= 0 and head_x < cnfg.WIDTH_TILES and head_y < cnfg.HEIGHT_TILES:
        h[int(head_y) + OFFSET, int(head_x) + OFFSET] = 1

    # add indicator for second body part. This is a critical feature.
    sb = np.zeros((cnfg.HEIGHT_TILES + PAD,cnfg.WIDTH_TILES + PAD))
    w = obs.body_parts[0]
    second_x = w[0]
    second_y = w[1]
    # second body part starts off the board
    if second_x >= 0 and second_y >= 0 and second_x < cnfg.WIDTH_TILES and second_y < cnfg.HEIGHT_TILES:
        sb[int(second_y) + OFFSET, int(second_x) + OFFSET] = 1

    # Tail
    t = np.zeros((cnfg.HEIGHT_TILES + PAD,cnfg.WIDTH_TILES + PAD))
    tail_x = obs.tail[0]
    tail_y = obs.tail[1]
    # Tail starts off the board
    if tail_x >= 0 and tail_y >= 0 and tail_x < cnfg.WIDTH_TILES and tail_y < cnfg.HEIGHT_TILES:
        t[int(tail_y) + OFFSET, int(tail_x) + OFFSET] = 1

    # Add indicators for each body part
    b = np.zeros((cnfg.HEIGHT_TILES + PAD,cnfg.WIDTH_TILES + PAD))
    for i,w in enumerate(obs.body_parts):

        if i > 0: # Second body part is it's own separate feature
            body_x = w[0]
            body_y = w[1]
            # Tail starts off the board
            if body_x >= 0 and body_y >= 0 and body_x < cnfg.WIDTH_TILES and body_y < cnfg.HEIGHT_TILES:
                b[int(body_y) + OFFSET, int(body_x) + OFFSET] = 1

    # Add 'ones' around perimeter
    topbottom = np.ones((1,cnfg.WIDTH_TILES + PAD))
    middle = np.zeros((1,cnfg.WIDTH_TILES + PAD))
    middle[0,0] = 1
    middle[0,-1] = 1
    middle = np.repeat(middle,cnfg.HEIGHT_TILES,axis=0)
    board = np.concatenate((topbottom,middle,topbottom),axis=0)

    f = np.stack((a,b,sb,h,t,board),2) #stack along the final dimension - needed for input into convolutional nn
    return f
    # return np.array(a + h + t + b)

def q_network(X_state, name):
    prev_layer = X_state
    with tf.variable_scope(name) as scope:
        for n_maps, kernel_size, strides, padding, activation in zip(
                cnfg.conv_n_maps, cnfg.conv_kernel_sizes, cnfg.conv_strides,
                cnfg.conv_paddings, cnfg.conv_activation):
            prev_layer = tf.layers.conv2d(
                prev_layer, filters=n_maps, kernel_size=kernel_size,
                strides=strides, padding=padding, activation=activation,
                kernel_initializer=initializer)
        last_conv_layer_flat = tf.reshape(prev_layer, shape=[-1, cnfg.n_hidden_in])

        hidden1 = tf.layers.dense(last_conv_layer_flat, cnfg.n_hidden1,
                                 activation=cnfg.hidden_activation,
                                 kernel_initializer=initializer)
        outputs = tf.layers.dense(hidden1, cnfg.n_outputs,
                                  kernel_initializer=initializer)
    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       scope=scope.name)
    trainable_vars_by_name = {var.name[len(scope.name):]: var
                              for var in trainable_vars}
    return outputs, trainable_vars_by_name

def epsilon_greedy(q_values, snakeLength, isOnEdge, suggestedAction, numGamesPlayed):
    #print(step)
    epsilon = .65
    if numGamesPlayed > 6000:
        epsilon = .5

    if numGamesPlayed > 12000:
        epsilon = .4

    if numGamesPlayed > 18000:
        epsilon = .3

    if numGamesPlayed > 24000:
        epsilon = .2

    if numGamesPlayed > 30000:
        epsilon = .15

    if numGamesPlayed > 40000:
        epsilon = .12

    if snakeLength > 10:
        epsilon = .1

    if snakeLength > 20:
        epsilon = .08

    if snakeLength > 30:
        epsilon = .05

    if snakeLength > 40:
        epsilon = .03

    #if isOnEdge:
    #    epsilon = min(epsilon, .15)

    # if step > 10000:
    #     epsilon = .05
    # epsilon = max(cnfg.eps_min, cnfg.eps_max - (cnfg.eps_max-cnfg.eps_min) * step/cnfg.eps_decay_steps)
    if np.random.rand() < epsilon:
        if np.random.rand() < 1.0 - cnfg.epsilon_guided:
            return np.random.randint(cnfg.n_outputs) # random action

        return suggestedAction # this is the move that will take snake closer to apple
    else:
        return np.argmax(q_values) # optimal action


initializer = tf.contrib.layers.variance_scaling_initializer()
X_state = tf.placeholder(tf.float32, shape=[None, cnfg.input_height, cnfg.input_width,
                                            cnfg.input_channels])
online_q_values, online_vars = q_network(X_state, name="q_networks/online")
target_q_values, target_vars = q_network(X_state, name="q_networks/target")

copy_ops = [target_var.assign(online_vars[var_name])
            for var_name, target_var in target_vars.items()]
copy_online_to_target = tf.group(*copy_ops)

X_action = tf.placeholder(tf.int32, shape=[None])
q_value = tf.reduce_sum(online_q_values * tf.one_hot(X_action, cnfg.n_outputs),
                        axis=1, keep_dims=True)

ytrain = tf.placeholder(tf.float32, shape=[None, 1])
error = tf.abs(ytrain - q_value)
clipped_error = tf.clip_by_value(error, 0.0, 1.0)
linear_error = 2 * (error - clipped_error)
loss = tf.reduce_mean(tf.square(ytrain - q_value))
# loss = tf.reduce_mean(tf.square(clipped_error) + linear_error)

global_step = tf.Variable(0, trainable=False, name='global_step')
# optimizer = tf.train.MomentumOptimizer(cnfg.learning_rate, cnfg.momentum, use_nesterov=True)
optimizer = tf.train.MomentumOptimizer(cnfg.learning_rate, cnfg.momentum)

training_op = optimizer.minimize(loss, global_step=global_step)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

# The next few lines are there for the purpose of being able to view things on tensorboard
# now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
# root_logdir = "tf_logs"
# logdir = "{}/run-{}/".format(root_logdir, now)
# file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
