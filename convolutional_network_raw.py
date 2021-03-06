""" Convolutional Neural Network.

Build and train a convolutional neural network with TensorFlow.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import division, print_function, absolute_import

import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./data/", one_hot=True)

# Training Parameters
learning_rate = 0.001
num_steps = 500
batch_size = 128
display_step = 10

# Network Parameters
num_input = 784  # MNIST data input (img shape: 28*28)
num_classes = 10  # MNIST total classes (0-9 digits)
dropout = 0.75  # Dropout, probability to keep units

# tf Graph input
X = tf.placeholder(tf.float32, [None, num_input])
Y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    # The stride of the sliding window for each
    #       dimension of `input`
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')
    # A list or tuple of 4 ints. The size of the window for each dimension
    # of the input tensor


# Create model
# conv1 =  (?, 28, 28, 32)
# maxpool1 =  (?, 14, 14, 32)
# conv2 =  (?, 14, 14, 64)
# maxpool2 =  (?, 7, 7, 64)
# fc1 =  (?, 3136)
# fc1_add =  (?, 1024)
# out =  (?, 10)
def conv_net(x, weights, biases, dropout):
    # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    print('conv1 = ', conv1.get_shape())
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)
    print('maxpool1 = ', conv1.get_shape())
    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    print('conv2 = ', conv2.get_shape())
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)
    print('maxpool2 = ', conv2.get_shape())
    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    print('fc1 = ', fc1.get_shape())
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    print('fc1_add = ', fc1.get_shape())
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    print('out = ', out.get_shape())
    return out


# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7 * 7 * 64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, num_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

# Construct model
logits = conv_net(X, weights, biases, keep_prob)
prediction = tf.nn.softmax(logits)  # 连续数值转化成相对概率

# Define loss and optimizer 封装了softmax
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# 控制要summary的变量
# Create a summary to monitor cost tensor
tf.summary.scalar("loss", loss_op)  # 必须要有添加监控的tensor
merged_summary_op = tf.summary.merge_all()

# Start training
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)

    # op to write logs to Tensorboard，控制是否添加graph,默认为none
    summary_writer = tf.summary.FileWriter("./logs/conv_raw/", sess.graph)

    for step in range(1, num_steps + 1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)

        # Run optimization op (backprop)
        _, summary = sess.run([train_op, merged_summary_op], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.8})
        # Write logs at every iteration
        summary_writer.add_summary(summary, step)

        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y,
                                                                 keep_prob: 1.0})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for 256 MNIST test images
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={X: mnist.test.images[:256],
                                                             Y: mnist.test.labels[:256],
                                                             keep_prob: 1.0}))
