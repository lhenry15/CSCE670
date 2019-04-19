import tensorflow as tf
import pandas as pd

df = pd.read_csv('./item_vector.csv')

num_input = 157
num_hidden = 50
num_code = 3
num_output = num_input
learning_rate = 0.01

X = tf.placeholder(tf.float32, shape=[None, num_input])

W1 = tf.Variable(tf.truncated_normal([num_input, num_hidden], stddev=0.1))
b1 = tf.Variable(tf.constant(0.1, shape=[num_hidden]))

hidden1 = tf.nn.relu(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.truncated_normal([num_hidden, num_code], stddev=0.1))
b2 = tf.Variable(tf.constant(0.1, shape=[num_code]))

code = tf.nn.relu(tf.matmul(hidden1, W2) + b2)

W3 = tf.Variable(tf.truncated_normal([num_code, num_hidden], stddev=0.1))
b3 = tf.Variable(tf.constant(0.1, shape=[num_hidden]))

hidden2 = tf.nn.relu(tf.matmul(code, W3) + b3)

W4 = tf.Variable(tf.truncated_normal([num_hidden, num_output], stddev=0.1))
b4 = tf.Variable(tf.constant(0.1, shape=[num_output]))

output = tf.nn.relu(tf.matmul(hidden2, W4) + b4)

loss = tf.reduce_mean(tf.square(output - X))
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

step = df.shape[1] * 3

with tf.Session() as sess:
    sess.run(init)

    for iteration in range(step):
        sess.run(train, feed_dict={X: df})

    embedding = code.eval(feed_dict={X: df})
    df_code = pd.DataFrame(embedding, index=df['item_id'], columns=['factor1', 'factor2', 'factor3'])
    df_code.to_csv('item_embedding.csv')