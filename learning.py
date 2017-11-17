import config as cnfg
import tensorflow as tf
import numpy as np
import os
from datetime import datetime


def sample_memories(replay_memory,batch_size):
    indices = np.random.permutation(len(replay_memory))[:batch_size]
    cols = [[]] * 5 # state, action, reward, next_state, continue
    for idx in indices:
        memory = replay_memory[idx]
        for col, value in zip(cols, memory):
            col.append(value)
    cols = [np.array(col) for col in cols]
    return (cols[0], cols[1], cols[2].reshape(-1, 1), cols[3],
            cols[4].reshape(-1, 1))

def update(gameHistory):
    # TODO: Reloading is very slow
    with tf.Session() as sess:
        if os.path.isfile(cnfg.checkpoint_path + ".index"):
            saver.restore(sess, cnfg.checkpoint_path)
        else:
            init.run()
            copy_online_to_target.run()
        step = global_step.eval()
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
        next_q_values = online_q_values.eval(
            feed_dict={X_state: X_next_state_val})
        max_next_q_values = np.max(next_q_values, axis=1, keepdims=True)
        y_val = rewards + continues * cnfg.discount_rate * max_next_q_values

        #for a,b,c,d in zip(y_val,X_state_val,X_action_val,continues):
        #    print(a)
        #    print(b)
        #    print(c)
        #    print(d)
        #    print('\n')
        # Train the online DQN
        #for i in range(300):
        q_values = online_q_values.eval(feed_dict={X_state: X_state_val})
            #print(q_values)
        training_op.run(feed_dict={X_state: X_state_val,
                                   X_action: X_action_val, ytrain: y_val})

        #variable_check_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                           #scope="q_networks/online")
        #for var in variable_check_list:
        #    print(var)
        #    print(var.eval())
        # Regularly copy the online DQN to the target DQN
        if step % cnfg.copy_steps == 0:
            copy_online_to_target.run()

        # And save regularly
        if step % cnfg.save_steps == 0:
            saver.save(sess, cnfg.checkpoint_path)

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


#convert game state into a feature vector
def preprocess_observation(obs):
    # Add indicators at each coordinate for apple
    a = [0] * cnfg.WIDTH_TILES * cnfg.HEIGHT_TILES
    apple_x = int(obs.apple[0])
    apple_y = int(obs.apple[1])
    a[width * apple_y + apple_x] = 1

    # Add indicators for head
    h = [0] * cnfg.WIDTH_TILES * cnfg.HEIGHT_TILES

    head_x = obs.head[0]
    head_y = obs.head[1]
    # If statement to account for when head goes off board
    if head_x >= 0 and head_y >= 0 and head_x < width and head_y < height:
        h[int(cnfg.WIDTH_TILES * head_y + head_x)] = 1

    # Tail
    t = [0] * cnfg.WIDTH_TILES * cnfg.HEIGHT_TILES
    tail_x = obs.tail[0]
    tail_y = obs.tail[1]
    t[int(cnfg.WIDTH_TILES * tail_y + tail_x)] = 1

    # Add indicators for each body part
    b = [0] * cnfg.WIDTH_TILES * cnfg.HEIGHT_TILES
    for w in obs.body_parts:
        body_x = w[0]
        body_y = w[1]
        b[int(cnfg.WIDTH_TILES * body_y + body_x)] = 1

    return np.array(a + h + t + b)

def q_network(X_state, name):
    prev_layer = X_state
    with tf.variable_scope(name) as scope:
        hidden = tf.layers.dense(prev_layer, cnfg.n_hidden,
                                 activation=cnfg.hidden_activation,
                                 kernel_initializer=initializer)
        outputs = tf.layers.dense(hidden, cnfg.n_outputs,
                                  kernel_initializer=initializer)
    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       scope=scope.name)
    trainable_vars_by_name = {var.name[len(scope.name):]: var
                              for var in trainable_vars}
    return outputs, trainable_vars_by_name

cnfg.hidden_activation = tf.nn.relu
initializer = tf.contrib.layers.variance_scaling_initializer()
X_state = tf.placeholder(tf.float32, shape=[None, cnfg.input_width])
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
loss = tf.reduce_mean(tf.square(clipped_error) + linear_error)

global_step = tf.Variable(0, trainable=False, name='global_step')
# optimizer = tf.train.momentumOptimizer(cnfg.learning_rate, cnfg.momentum, use_nesterov=True)
optimizer = tf.train.momentumOptimizer(cnfg.learning_rate, cnfg.momentum)
training_op = optimizer.minimize(loss, global_step=global_step)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

# The next few lines are there for the purpose of being able to view things on tensorboard
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
        return np.random.randint(cnfg.n_outputs) # random action
    else:
        return np.argmax(q_values) # optimal action
