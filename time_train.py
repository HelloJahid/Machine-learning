import numpy as np
import pandas as pd
import tensorflow as tf
import time

# read data from csv file
train_csv = pd.read_csv(r"C:\Users\JAHID\Desktop\ML\Project base\Time Series\train_data.csv")
all_cols = ['ID', 'Datetime', 'Count']

# convert csv data to pandas DataFrame
train_df = pd.DataFrame(train_csv)

# Handle the missing value
train_df = train_df.interpolate()  # fill with meadian value, apply only for float or int type data
train_df = train_df.fillna(method='bfill')  # fill with previous value , this time apply for categorical data


# time Series data
TS = np.array(train_df['Count'])
num_periods = 20
f_horizon = 1

x_data = TS[: len(TS) - (len(TS) % num_periods)]
x_batches = x_data.reshape(-1, 20, 1)
print(x_data.shape)

y_data = TS[1 : (len(TS) - (len(TS) % num_periods)) + f_horizon ]
y_batches = y_data.reshape(-1, 20, 1)

n_predict_data = 5120

def test_data(series, forecast, num_periods):
    test_x_setup = TS[-(n_predict_data + forecast) :]
    testX = test_x_setup[: n_predict_data].reshape(-1, 20, 1)
    testY = TS[-(num_periods) : ].reshape(-1, 20, 1)
    return testX, testY

X_test, Y_test = test_data(TS, f_horizon, num_periods)


## Tensorflow
n_neurons = 100
n_steps = 20
n_inputs = 1
n_outputs = 1

X = tf.placeholder(tf.float32, shape=[None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, shape=[None, n_steps, n_outputs])

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)

'''
# apply dropout
keep_prob = 0.5
basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
cells_drop = tf.contrib.rnn.DropoutWrapper(basic_cell, input_keep_prob=keep_prob)

rnn_outputs, states = tf.nn.dynamic_rnn(cells_drop, X, dtype=tf.float32)

stacked_rnn_output = tf.reshape(rnn_outputs, [-1, n_neurons])
stacked_outputs = tf.layers.dense(stacked_rnn_output, n_outputs)
outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_outputs])

'''

rnn_outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)

stacked_rnn_output = tf.reshape(rnn_outputs, [-1, n_neurons])
stacked_outputs = tf.layers.dense(stacked_rnn_output, n_outputs)
outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_outputs])



loss = tf.reduce_mean(tf.square(outputs - y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
training_op = optimizer.minimize(loss)
init = tf.global_variables_initializer()

epochs = 50001

with tf.Session() as sess:
    init.run()
    for ep in range(epochs):
        sess.run(training_op, feed_dict={X:x_batches, y:y_batches})
        if ep % 100 == 0:
            mse = loss.eval(feed_dict={X:x_batches, y:y_batches})
            print(ep, "\tMSE : ",mse)

    y_pred = sess.run(outputs, feed_dict={X:X_test})
    y_pred = y_pred.reshape(n_predict_data, -1)
    print(y_pred)

new_df = pd.DataFrame(y_pred , columns=['Count'])

print(new_df.head())

new_df.to_csv(r'C:\Users\JAHID\Desktop\ML\project base\data.csv')
