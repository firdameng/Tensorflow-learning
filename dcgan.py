""" Deep Convolutional Generative Adversarial Network (DCGAN).

Using deep convolutional generative adversarial networks (DCGAN) to generate
digit images from a noise distribution.

References:
    - Unsupervised representation learning with deep convolutional generative
    adversarial networks. A Radford, L Metz, S Chintala. arXiv:1511.06434.

Links:
    - [DCGAN Paper](https://arxiv.org/abs/1511.06434).
    - [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import division, print_function, absolute_import

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./data/", one_hot=True)

# Training Params
num_steps = 20000
batch_size = 32

# Network Params
image_dim = 784 # 28*28 pixels * 1 channel
gen_hidden_dim = 256
disc_hidden_dim = 256
noise_dim = 200 # Noise data points


# Generator Network
# Input: Noise, Output: Image
# gen x =  (?, 200)
# gen dense1 =  (?, 4608)
# gen reshape1 =  (?, 6, 6, 128)
# gen conv2d_transpose 1 =  (?, 14, 14, 64)
# gen conv2d_transpose 2 =  (?, 28, 28, 1)
def generator(x, reuse=False):
    with tf.variable_scope('Generator', reuse=reuse):
        # TensorFlow Layers automatically create variables and calculate their
        # shape, based on the input.
        print('gen x = ', x.get_shape())
        x = tf.layers.dense(x, units=6 * 6 * 128)
        print('gen dense1 = ', x.get_shape())
        x = tf.nn.tanh(x)
        # Reshape to a 4-D array of images: (batch, height, width, channels)
        # New shape: (batch, 6, 6, 128)
        x = tf.reshape(x, shape=[-1, 6, 6, 128])
        print('gen reshape1 = ', x.get_shape())   # 由逆过程去推断
        # Deconvolution, image shape: (batch, 14, 14, 64)
        x = tf.layers.conv2d_transpose(x, 64, 4, strides=2)
        print('gen conv2d_transpose 1 = ', x.get_shape())
        # Deconvolution, image shape: (batch, 28, 28, 1)
        x = tf.layers.conv2d_transpose(x, 1, 2, strides=2)
        print('gen conv2d_transpose 2 = ', x.get_shape())
        # Apply sigmoid to clip values between 0 and 1
        x = tf.nn.sigmoid(x)
        return x


# Discriminator Network
# Input: Image, Output: Prediction Real/Fake Image
# disc x =  (?, 28, 28, 1)
# disc conv1=  (?, 24, 24, 64)
# disc average_pooling2d 1 =  (?, 12, 12, 64)
# disc conv2=  (?, 8, 8, 128)
# disc average_pooling2d 2 =  (?, 4, 4, 128)
# disc flatten =  (?, 2048)
# disc dense1 =  (?, 1024)
# disc dense2 =  (?, 2)
def discriminator(x, reuse=False):
    with tf.variable_scope('Discriminator', reuse=reuse):
        # Typical convolutional neural network to classify images.
        print('disc x = ', x.get_shape())
        x = tf.layers.conv2d(x, 64, 5)                  # 有效卷积，s = 1
        print('disc conv1= ', x.get_shape())
        x = tf.nn.tanh(x)
        x = tf.layers.average_pooling2d(x, 2, 2)        # 无参数，但也特征图减少
        print('disc average_pooling2d 1 = ', x.get_shape())
        x = tf.layers.conv2d(x, 128, 5)
        print('disc conv2= ', x.get_shape())
        x = tf.nn.tanh(x)
        x = tf.layers.average_pooling2d(x, 2, 2)
        print('disc average_pooling2d 2 = ', x.get_shape())
        x = tf.contrib.layers.flatten(x)
        print('disc flatten = ', x.get_shape())
        x = tf.layers.dense(x, 1024)
        print('disc dense1 = ', x.get_shape())
        x = tf.nn.tanh(x)
        # Output 2 classes: Real and Fake images
        x = tf.layers.dense(x, 2)
        print('disc dense2 = ', x.get_shape())
    return x

# Build Networks
# Network Inputs
noise_input = tf.placeholder(tf.float32, shape=[None, noise_dim])
real_image_input = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])

# Build Generator Network
gen_sample = generator(noise_input)

# Build 2 Discriminator Networks (one from noise input, one from generated samples)
disc_real = discriminator(real_image_input)
disc_fake = discriminator(gen_sample, reuse=True)        # 共享判别器的参数
disc_concat = tf.concat([disc_real, disc_fake], axis=0)  # 两行，低纬度拼接

# Build the stacked generator/discriminator   其实就是g+d连接到一起
stacked_gan = discriminator(gen_sample, reuse=True)

# Build Targets (real or fake images)  需要喂养的数据
disc_target = tf.placeholder(tf.int32, shape=[None])
gen_target = tf.placeholder(tf.int32, shape=[None])

# Build Loss
disc_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=disc_concat, labels=disc_target))
gen_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=stacked_gan, labels=gen_target))

# Build Optimizers
optimizer_gen = tf.train.AdamOptimizer(learning_rate=0.001)
optimizer_disc = tf.train.AdamOptimizer(learning_rate=0.001)

# Training Variables for each optimizer
# By default in TensorFlow, all variables are updated by each optimizer, so we
# need to precise for each one of them the specific variables to update.
# Generator Network Variables  只获取g域下的变量了
gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
# Discriminator Network Variables
disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')

# Create training operations   这一步就是后续sess.run进行的训练操作
train_gen = optimizer_gen.minimize(gen_loss, var_list=gen_vars)
train_disc = optimizer_disc.minimize(disc_loss, var_list=disc_vars)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for i in range(1, num_steps+1):

        # Prepare Input Data
        # Get the next batch of MNIST data (only images are needed, not labels)
        batch_x, _ = mnist.train.next_batch(batch_size)
        batch_x = np.reshape(batch_x, newshape=[-1, 28, 28, 1])
        # Generate noise to feed to the generator
        z = np.random.uniform(-1., 1., size=[batch_size, noise_dim])

        # Prepare Targets (Real image: 1, Fake image: 0)
        # The first half of data fed to the generator are real images,
        # the other half are fake images (coming from the generator).
        batch_disc_y = np.concatenate(
            [np.ones([batch_size]), np.zeros([batch_size])], axis=0)
        # Generator tries to fool the discriminator, thus targets are 1.
        batch_gen_y = np.ones([batch_size])

        # Training
        feed_dict = {real_image_input: batch_x, noise_input: z,
                     disc_target: batch_disc_y, gen_target: batch_gen_y}
        _, _, gl, dl = sess.run([train_gen, train_disc, gen_loss, disc_loss],
                                feed_dict=feed_dict)
        if i % 100 == 0 or i == 1:
            print('Step %i: Generator Loss: %f, Discriminator Loss: %f' % (i, gl, dl))

    # Generate images from noise, using the generator network.
    f, a = plt.subplots(4, 10, figsize=(10, 4))
    for i in range(10):
        # Noise input.
        z = np.random.uniform(-1., 1., size=[4, noise_dim])
        g = sess.run(gen_sample, feed_dict={noise_input: z})  # 不保证生成什么鬼
        for j in range(4):
            # Generate image from noise. Extend to 3 channels for matplot figure.
            img = np.reshape(np.repeat(g[j][:, :, np.newaxis], 3, axis=2),
                             newshape=(28, 28, 3))
            a[j][i].imshow(img)

    f.show()
    plt.draw()
    plt.waitforbuttonpress()
