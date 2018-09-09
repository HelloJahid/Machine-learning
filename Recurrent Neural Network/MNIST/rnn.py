from sklearn.datasets import fetch_mldata
import tensorflow as tf
import numpy as np
import time

# =================================== Prepare the datasets =========================

# import MNIST datasets
mnist = fetch_mldata('MNIST original')

# split the MNIST data into faeture and target
X = mnist['data']  # feature
y = mnist['target']  # target


# split the the feature data into Train and Test set
X_train , X_test = X[:60000], X[60000:]

# split the the target data into Train and Test set
y_train, y_test = y[:60000], y[60000:]

# numpy shuffle, use to shuffle any index
shuffle_index_train = np.random.permutation(60000)

# shuffle the index of Training data for better performance
X_train, y_train = X_train[shuffle_index_train], y_train[shuffle_index_train]



# =================================  Tensorflow Operation   ===================


# required data
n_steps = 28  # input sequence
n_inputs = 28  # number input
n_neurons = 150 # number of recurrent neuron
n_outputs = 10  # output of each class
learning_rate = 0.001 # learning rate of gradient descent

### placeholder
X = tf.placeholder(tf.float32, [None, n_steps, n_inputs]) # input
y = tf.placeholder(tf.int32, [None]) # output

# RNN Operation
basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu)
outputs , states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)

# determine loss of the network
logits = tf.layers.dense(states, n_outputs)
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
loss = tf.reduce_mean(xentropy)

# optimize the loss
training_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# compute accuracy
correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))


# create batch for Training iteration
def shuffle_index(X, y, batch_size):
    rand_indx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_indx in np.array_split(rand_indx, n_batches):
        X_batch, y_batch = X[batch_indx], y[batch_indx]
        yield X_batch, y_batch



init = tf.global_variables_initializer()
num_epochs = 100
batch_size = 150

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(num_epochs):
        start_time = time.time()

        for X_batch, y_batch in shuffle_index(X_train, y_train, batch_size):
            X_batch = X_batch.reshape(-1, n_steps, n_inputs)
            sess.run(training_op, feed_dict={X:X_batch, y:y_batch})

        acc_train = accuracy.eval(feed_dict={X:X_batch, y:y_batch})
        end_time = time.time()
        print("Epoch "+str(epoch+1)+" completed : Time usage "+str(int(end_time-start_time))+" seconds")
        print("\tAccuracy:")
        print ("\t- Training Accuracy:\t{}".format(acc_train))



