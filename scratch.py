import tensorflow as tf
import numpy as np


A = tf.constant([[[10,20,30], [40,50,60], [70,80,90]], [[11,21,31], [41,51,61], [71,81,91]]])

print("A=\n", A)

new_shape = [2*3,3]
A_reshaped = tf.reshape(A, new_shape)
print("A_reshaped=\n", A_reshaped)

A_tall = tf.reshape(A, [-1])
print("A_tall=\n", A_tall)

A_reconstructed = tf.reshape(A_tall, A.shape)
print("A_reconstructed=\n", A_reconstructed)

print("done")