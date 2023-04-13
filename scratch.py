import tensorflow as tf
import numpy as np


# Create a 3 x 3 Toeplitz operator.
# col = [1., 2., 3.]
# row = [1., 0., 0.]
# operator = tf.linalg.LinearOperatorToeplitz(col, row)
# len = 11520*8
# tensor1 = tf.random.uniform(shape=(len,), dtype=tf.float32)
# tensor1 = tf.complex(tensor1, tf.random.uniform(shape=(len,), dtype=tf.float32))
# # Create a second 1D tensor of length 11520 and data type complex with zeros except for the first element.
# tensor2 = tf.concat([tf.expand_dims(tensor1[0], 0), tf.zeros(shape=(len-1,), dtype=tf.complex64)], axis=0)
# operator = tf.linalg.LinearOperatorToeplitz(tensor1, tensor2)
# # print(operator.to_dense())
# # np.zeros((len, len), dtype=np.complex64)
# tensor3 = tf.reshape(tensor1, shape=(len, 1))
# result = operator.matmul(tensor3)

G_tensor = tf.zeros([5,3],dtype=tf.complex64)
G_tensor_dash = tf.zeros([3,5],dtype=tf.complex64)
updates = tf.constant([1,2,3,1,2,3,1,2,3], dtype=tf.complex64)

# indices = tf.where(tf.linalg.band_part(tf.ones_like(G_tensor), 2, 0))
indices = tf.where(tf.linalg.band_part(tf.ones_like(G_tensor_dash), 0, 2))
indices = tf.stack([indices[:, 1], indices[:, 0]], axis=1)
# num_indices = tf.shape(indices)[0]
# updates = tf.reshape(updates[:num_indices], shape=(-1, 1))

print("indices:\n", indices)

G_tensor = tf.tensor_scatter_nd_update(G_tensor, indices, updates)

print("G_tensor:\n", G_tensor)


# # Define a non-square matrix
# A = tf.ones([5, 3])

# # Keep only the lower triangle
# B = tf.linalg.band_part(A, num_lower=2, num_upper=0)

# # Print the original and resulting matrices
# print("A:\n", A)
# print("B:\n", B)


print("done")