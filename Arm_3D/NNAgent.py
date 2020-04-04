# Librerias necesarias
import tensorflow as tf
import math
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque

# Creamos la Clase QNetwork en donde se define el modelo de la red
class QNetwork:
    def __init__(self, learning_rate=0.01, state_size=4, action_size=2, hidden_size=10, batch_size=20, name='QNetwork'):
        # Estados de entrada a la red neuronal
        with tf.compat.v1.variable_scope(name):
            self.inputs_ = tf.compat.v1.placeholder(
                tf.float32, [None, state_size], name='inputs')
            self.actions_ = tf.compat.v1.placeholder(
                tf.int32, [batch_size], name='actions')
            one_hot_actions = tf.one_hot(self.actions_, action_size)

            # Target Q values for training
            self.targetQs_ = tf.compat.v1.placeholder(tf.float32, [None], name='target')

            # Capas ocultas (Tanh)
            self.fc1 = tf.contrib.layers.fully_connected(
                self.inputs_, hidden_size, activation_fn=tf.nn.tanh)
            self.fc2 = tf.contrib.layers.fully_connected(
                self.fc1, hidden_size, activation_fn=tf.nn.tanh)

            # Capa lineal de salida
            self.output = tf.contrib.layers.fully_connected(
                self.fc2, action_size, activation_fn=None)

            # Train with loss (targetQ - Q)^2
            # output has length 2, for two actions. This next line chooses
            # one value from output (per row) according to the one-hot encoded actions.
            self.Q = tf.reduce_sum(tf.multiply(
                self.output, one_hot_actions), axis=1)

            self.loss = tf.reduce_mean(tf.square(self.targetQs_ - self.Q))
            self.opt = tf.compat.v1.train.AdamOptimizer(
                learning_rate).minimize(self.loss)


class Memory():
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)),
                               size=batch_size,
                               replace=False)
        return [self.buffer[ii] for ii in idx]
