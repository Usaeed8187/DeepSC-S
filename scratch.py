import tensorflow as tf
import numpy as np


# Create a 3 x 3 Toeplitz operator.
# col = [1., 2., 3.]
# row = [1., 0., 0.]
# operator = tf.linalg.LinearOperatorToeplitz(col, row)

len = 11520*8

tensor1 = tf.random.uniform(shape=(len,), dtype=tf.float32)
tensor1 = tf.complex(tensor1, tf.random.uniform(shape=(len,), dtype=tf.float32))

# Create a second 1D tensor of length 11520 and data type complex with zeros except for the first element.
tensor2 = tf.concat([tf.expand_dims(tensor1[0], 0), tf.zeros(shape=(len-1,), dtype=tf.complex64)], axis=0)

operator = tf.linalg.LinearOperatorToeplitz(tensor1, tensor2)

# print(operator.to_dense())

# np.zeros((len, len), dtype=np.complex64)

tensor3 = tf.reshape(tensor1, shape=(len, 1))

result = operator.matmul(tensor3)

print("done")