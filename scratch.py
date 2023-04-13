import tensorflow as tf
import numpy as np


# Input parameters
l_max = 2
a = np.array([[1],[2],[3]])
gs = np.tile(a,(1,9))
MbyN = 9
r = np.zeros([MbyN + l_max], dtype=complex)
s = [1,2,3,4,5,6,7,8,9]

# Loop convolution operation
for l in range(MbyN): # loop over transmit signal and apply time-varying channel on it
    r[l:(l + l_max + 1)] += s[l] * gs[:, l]
# r[:l_max] += r[-l_max:] # effect of channel on cyclic prefix
r = r[:-l_max] # remove excess delay (handled by removing cyclic prefix of next OTFS grid)
print(r)


# Tensor multiplication operation
rows = MbyN + l_max + 1
cols = MbyN
tensor_shape = (rows, cols)
G_tensor = tf.zeros(tensor_shape)
indices = tf.where(tf.linalg.band_part(tf.ones([cols,rows]), 0, l_max))
indices = tf.reverse(indices, axis=[1])
updates = tf.constant(gs.flatten('F'))
G_tensor = tf.tensor_scatter_nd_update(G_tensor, indices, tf.cast(updates, G_tensor.dtype))
s_tensor = tf.expand_dims(s, axis=1)
s_tensor = tf.cast(s_tensor, G_tensor.dtype)
r_tensor = tf.matmul(G_tensor, s_tensor)
r_tensor = r_tensor[:cols]

print(r_tensor)

print("done")