import config
import tensorflow as tf
import numpy as np
import os

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

def update(gameHistory):
    with tf.Session() as sess:
        if os.path.isfile(checkpoint_path + ".index"):
            saver.restore(sess, checkpoint_path)
        else:
            init.run()
            copy_online_to_target.run()
        step = global_step.eval()
        X_state_val, X_action_val, rewards, X_next_state_val, continues = sample_memories(gameHistory,batch_size)

        #if len(gameHistory) < 100: #or iteration % training_interval != 0:
        #   return

        # Sample memories and use the target DQN to produce the target Q-Value
        #X_state_val, X_action_val, rewards, X_next_state_val, continues = (
        #    sample_memories(batch_size))
        #X_next_state_val = np.array(X_next_state_val).reshape(1,input_width)
        #X_state_val = np.array(X_state_val).reshape(1,input_width)
        #X_action_val = np.array(X_action_val).reshape(1)
        #print(X_state_val, X_action_val, 'rewards', rewards, X_next_state_val, 'continues', continues )
        next_q_values = online_q_values.eval(
            feed_dict={X_state: X_next_state_val})
        max_next_q_values = np.max(next_q_values, axis=1, keepdims=True)
        y_val = rewards + continues * discount_rate * max_next_q_values

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

gwidth = config.DEFAULT_WINDOW_WIDTH/config.STEP_SIZE
gheight = config.DEFAULT_WINDOW_HEIGHT/config.STEP_SIZE
n_hidden = 20
input_width = int(gwidth * gheight * 3)
hidden_activation = None
n_outputs = 4  # 4 discrete actions are available

learning_rate = 0.001
momentum = 0.9


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

    #need an extra set of parameters for head location so game can figure out
    #where head actually is

    c = [0] * width * height
    x = obs.head[0]
    y = obs.head[1]
    #if statement to account for when head goes off board
    if x >= 0 and y >= 0 and x <width and y < height:
        c[int(width * y + x)] = 1
    return np.array(a + b + c)

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
#optimizer = tf.train.MomentumOptimizer(learning_rate, momentum, use_nesterov=True)
optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
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
        return np.argmax(q_values) # optimal action

n_steps = 4000000  # total number of training steps
training_start = 10000  # start training after 10,000 game iterations
training_interval = 4  # run a training step every 4 game iterations
save_steps = 1  # save the model every 1,000 training steps
copy_steps = 1  # copy online DQN to target DQN every 10,000 training steps
discount_rate = 0.99
skip_start = 90  # Skip the start of every game (it's just waiting time).
batch_size = 20
iteration = 0  # game iterations
checkpoint_path = "./my_dqn.ckpt"
done = True # env needs to be reset
