import numpy as np
import tensorflow as tf

class channel_generator_fractional:
    # np_config.enable_numpy_behavior()
    def __init__(self, OTFS_para):

        # sess = tf.compat.v1.Session()
        # OTFS_para_value = sess.run(OTFS_para)

        self.N_r = 1
        self.N_t = 1
        self.M = OTFS_para['N_c']
        self.N = OTFS_para['N_slot']
        self.car_fre = OTFS_para['car_fre']
        self.delta_f = OTFS_para['delta_f']
        self.max_speed = OTFS_para['mobility_speed']
        # EVA channel model
        self.delays = np.array([0, 30, 150, 310, 370, 710, 1090, 1730, 2510]) * 10 ** (-9)
        self.pdp = np.array([0, -1.5, -1.4, -3.6, -0.6, -9.1, -7.0, -12.0, -16.9])

    def generate_delay_doppler_channel_param(self):
        T = 1 / self.delta_f
        taps = len(self.delays)
        one_delay_tap = 1 / (self.M * self.delta_f)
        one_doppler_tap = 1 / (self.N * T)
        delay_taps = np.round(self.delays / one_delay_tap).astype(int)

        pow_prof = 10 ** (self.pdp / 10)
        pow_prof = pow_prof / np.sum(pow_prof) # normalization of power delay profile
        chan_coef = np.sqrt(pow_prof) * np.sqrt(0.5) * (np.random.randn(taps) + 1j * np.random.randn(taps))

        max_UE_speed = self.max_speed * 1000 / 3600
        doppler_vel = max_UE_speed * self.car_fre / (3 * 10**8)
        max_doppler_tap = doppler_vel / one_doppler_tap
        doppler_taps = max_doppler_tap * np.cos(2*np.pi*np.random.rand(taps))

        return chan_coef, delay_taps, doppler_taps, taps

    def gen_discrete_time_channel(self, chan_coef, delay_taps, doppler_taps, taps):
        frame_size = self.N * self.M
        z = np.exp(1j * 2 * np.pi / self.N / self.M)
        l_max = max(delay_taps)
        gs = np.zeros([l_max + 1, frame_size], dtype=complex)
        for q in range(frame_size):
            for i in range(taps):
                l_i = delay_taps[i]
                k_i = doppler_taps[i]
                gs[l_i, q] = gs[l_i, q] + chan_coef[i] * z ** (k_i * (q - l_i))

        return gs

    def otfs_channel_output(self, delay_taps, gs, s):

        l_max = max(delay_taps)

        # r = np.zeros([self.N * self.M + l_max], dtype=complex)
        
        # r = tf.dtypes.complex(real=r_real, imag=r_imag)
        # r_real = tf.zeros([self.N * self.M + l_max, 1])
        # r_imag = tf.zeros([self.N * self.M + l_max, 1])

        # tensor_shape = [self.N * self.M + l_max, 1]
        # r = tf.Variable(tf.zeros(shape=tensor_shape, dtype=tf.complex64))

        # gs_tensor = tf.constant(gs, dtype=tf.complex64)

        # for l in range(self.N * self.M):
        #     indices = [i for i in range(l, (l + l_max + 2))]
        #     values = s[l] * gs_tensor[:, l]
        #     for idx, val in zip(indices, values):
        #         r = tf.tensor_scatter_nd_update(r, [idx], [val])
        #     # r[l:(l + l_max + 1)] += s[l] * gs_tensor[:, l]
        # r[:l_max] += r[-l_max:] NOTE: not 100% sure if this is correct
        # r = r[:-l_max]

        # G = np.zeros((self.M*self.N+ l_max, self.M*self.N), dtype=np.complex64)

        rows = self.M*self.N+ l_max + 1
        cols = self.M*self.N
        tensor_shape = (rows, cols)
        G_tensor = tf.zeros(tensor_shape, dtype=tf.complex64)

        indices = tf.where(tf.linalg.band_part(tf.ones([cols,rows]), 0, l_max))

        updates = tf.constant(gs.flatten(),dtype=tf.complex64)

        G_tensor = tf.tensor_scatter_nd_update(G_tensor, indices, updates)

        # for i in range(self.N * self.M):
        #     row_indices = tf.range(i, l_max + 1 + i, dtype=tf.int64)
        #     column_index = tf.constant([i] * (l_max + 1), dtype=tf.int64)
        #     indices = tf.stack([row_indices, column_index], axis=1)
        #     # indices = tf.constant([[j, i] for j in range(i, l_max + 1 + i)])
        #     updates = tf.constant(gs[:, i], dtype=tf.complex64)
        #     G_tensor = tf.tensor_scatter_nd_update(G_tensor, indices, updates)

            # G[i:l_max+1+i,i] = gs[:,i] 
        
        # G[:l_max,:] += G[-l_max:,:] #effect of multipath on the first l_max symbols. NOTE: matches line 74. 
        # G = G[:-l_max,:]

        # G_tensor = tf.constant(G)
        # s_reshaped = tf.reshape(s, [self.M, 1])
        s = tf.cast(s, G_tensor.dtype)
        s = tf.expand_dims(s, axis=1)
        r = tf.matmul(G_tensor, s)
        r = r[:cols]


        # G_tensor = 

        return r
