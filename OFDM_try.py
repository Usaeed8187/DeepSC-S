import tensorflow as tf
import numpy as np

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

CP = 64
pilotSymbols = 4


def ToOFDM(x_complex, batch_size=32):

    # Reshape to (batch_size * num_of_subframes * num_of_symbols * num_of_subcarrieres)
    num_symbol = 16
    num_carrier = 512
    x_ofdm = tf.reshape(x_complex, [batch_size, -1, num_symbol, num_carrier])
    x_ofdm = tf.cast(x_ofdm, tf.complex128)

    # Add pilots
    shape_p = x_ofdm.get_shape().as_list()

    pilotReal = tf.random.normal(shape=(pilotSymbols, shape_p[3]))
    pilotImag = tf.random.normal(shape=(pilotSymbols, shape_p[3]))

    pilotReal = tf.tile(pilotReal[tf.newaxis, tf.newaxis, :, :],
                        multiples=[shape_p[0], shape_p[1], 1, 1])

    pilotImag = tf.tile(pilotImag[tf.newaxis, tf.newaxis, :, :],
                        multiples=[shape_p[0], shape_p[1], 1, 1])

    pilotValue = tf.dtypes.complex(real=pilotReal, imag=pilotImag)

    pilotValue = tf.cast(pilotValue, tf.complex128)

    x_ofdm = tf.concat([pilotValue, x_ofdm], axis=2)

    # IFFT
    x_t = tf.signal.ifft(x_ofdm)

    # Add CP
    cp_t = x_t[..., -CP:]
    cp_t = tf.cast(cp_t, tf.complex128)
    x_t = tf.concat([cp_t, x_t], axis=-1)

    # Parallel to Serial
    shape_s = x_t.get_shape().as_list()
    x_t = tf.reshape(x_t, [shape_s[0], shape_s[1], -1])

    return x_t, shape_s


def FromOFDM(x_t, shape_s=[32, 8, 20, 576], _shape=[32, 128, 32, 32]):

    batch_size = shape_s[0]
    x_t = tf.reshape(x_t, shape_s)

    # Remove CP
    x_t = x_t[..., CP:]

    # FFT
    x_f = tf.signal.fft(x_t)

    # Extract pilots and data
    pilot = x_f[..., 0:pilotSymbols, :]
    x_f = x_f[..., pilotSymbols:, :]

    # Reshape back to the original input size
    x_after = tf.reshape(x_f, [batch_size, _shape[1], _shape[2]*_shape[3]//2])
    x_after = tf.cast(x_after, tf.complex64)

    return x_after, pilot


# Just for the test of OFDM implementation
if __name__ == "__main__":

    # Assume the input size is  (batch size)32 * 32 * 32 * 128
    _input = tf.random.normal(shape=(32, 32, 32, 128))

    _input = tf.transpose(_input, perm=[0, 3, 1, 2])

    batch_size = tf.shape(_input)[0]
    _shape = _input.get_shape().as_list()

    _shape = _input.get_shape().as_list()

    x = tf.reshape(_input, [batch_size, _shape[1], _shape[2]*_shape[3]//2, 2])

    x_norm = tf.math.sqrt(_shape[2]*_shape[3]//2 / 2.0) * \
        tf.math.l2_normalize(x, axis=2)

    print(x.shape)

    x_real = x_norm[:, :, :, 0]
    x_imag = x_norm[:, :, :, 1]
    x_complex = tf.dtypes.complex(real=x_real, imag=x_imag)

    print(x_complex.shape)

    # Reshape to (batch_size * num_of_subframes * num_of_symbols * num_of_subcarrieres)
    num_symbol = 16
    num_carrier = 512
    x_ofdm = tf.reshape(x_complex, [batch_size, -1, num_symbol, num_carrier])

    # Have to cast it to complex128.
    # Otherwise, the recovered inputs will change after the whole process
    x_ofdm = tf.cast(x_ofdm, tf.complex128)
    print(x_ofdm.shape)

    x_data = x_ofdm

    # Add pilots
    shape_p = x_ofdm.get_shape().as_list()
    pilotSymbols = 4

    pilotReal = tf.random.normal(shape=(pilotSymbols, shape_p[3]))
    pilotImag = tf.random.normal(shape=(pilotSymbols, shape_p[3]))

    pilotReal = tf.tile(pilotReal[tf.newaxis, tf.newaxis, :, :],
                        multiples=[shape_p[0], shape_p[1], 1, 1])

    pilotImag = tf.tile(pilotImag[tf.newaxis, tf.newaxis, :, :],
                        multiples=[shape_p[0], shape_p[1], 1, 1])

    pilotValue = tf.dtypes.complex(real=pilotReal, imag=pilotImag)

    pilotValue = tf.cast(pilotValue, tf.complex128)

    print(pilotValue.shape)

    x_ofdm = tf.concat([pilotValue, x_ofdm], axis=2)

    print("OFDM: {}".format(x_ofdm.shape))

    # IFFT
    x_t = tf.signal.ifft(x_ofdm)
    print("Time Domain: {}".format(x_t.shape))

    x_cp = x_t

    # Add CP
    CP = 64
    cp_t = x_t[..., -CP:]
    cp_t = tf.cast(cp_t, tf.complex128)
    x_t = tf.concat([cp_t, x_t], axis=-1)
    print("After CP: {}".format(x_t.shape))

    x_ps = x_t

    # Parallel to Serial
    shape_s = x_t.get_shape().as_list()
    x_t = tf.reshape(x_t, [shape_s[0], shape_s[1], -1])
    print("P2S: {}".format(x_t.shape))

    p2s = tf.zeros([0])
    p2s = tf.cast(p2s, tf.complex128)
    for i in range(shape_s[-2]):
        p2s = tf.concat([p2s, x_ps[0, 0, i, :]], axis=0)

    res = tf.reduce_all(tf.equal(p2s, x_t[0, 0, :]))
    print("*****P2S reshaping is correct: {}*****".format(res))

    # Transmit through the channel

    shape_n = x_t.get_shape().as_list()
    shape_n.append(2)
    h = tf.random.normal(shape=shape_n, dtype=tf.float32)
    h = (tf.math.sqrt(1./2.) + tf.math.sqrt(1./2.)*h) / tf.math.sqrt(2.)
    # h = h / tf.math.sqrt(2.0)
    h_real = h[..., 0]
    h_imag = h[..., 1]
    h_complex = tf.dtypes.complex(real=h_real, imag=h_imag)
    #h_complex = tf.reshape(h_complex, [shape_s[0], shape_s[1], -1])
    h_complex = tf.cast(h_complex, tf.complex128)

    db = 10
    y_complex = tf.math.multiply(h_complex, x_t)
    squ = tf.square(tf.abs(y_complex))
    sig_power = tf.math.reduce_mean(squ, axis=[2,1,0])
    std = sig_power * 10**(-db/10) 
    # std = tf.tile(std[:,tf.newaxis,tf.newaxis],
    #               multiples=[1, shape_n[-3], shape_n[-2]])
    std = tf.sqrt(std/2)
    std = tf.cast(std, tf.float32)
    print(std)

    n = tf.random.normal(shape=shape_n, mean=0.0, stddev=std,dtype=tf.float32)
    n_real = n[..., 0]
    n_imag = n[..., 1]
    n_complex = tf.dtypes.complex(real=n_real, imag=n_imag)
    n_complex = tf.cast(n_complex, tf.complex128)
    #n_complex = tf.reshape(n_complex, [shape_s[0], shape_s[1], -1])
    # n_complex = tf.math.multiply(n_complex, std)

    # Channel End


    # Serial to Parallel
    x_t = tf.reshape(x_t, shape_s)
    print("S2P: {}".format(x_t.shape))

    res = tf.reduce_all(tf.equal(x_ps, x_t))
    print("Data after P2S and S2P do not change: {}".format(res))

    # Remove CP
    x_t = x_t[..., CP:]
    res = tf.reduce_all(tf.equal(x_cp, x_t))
    print("Data after adding and removing CP do not change: {}".format(res))

    # FFT
    x_f = tf.signal.fft(x_t)
    res = tf.reduce_all(tf.abs(x_ofdm - x_f) < 1e-6)
    print("Symbols after IFFT and FFT do not change: {}".format(res))

    # Extract pilots and data
    pilot = x_f[..., 0:pilotSymbols, :]
    x_f = x_f[..., pilotSymbols:, :]

    res = tf.reduce_all(tf.abs(pilotValue - pilot) < 1e-6)
    print("Pilots after IFFT and FFT do not change: {}".format(res))

    res = tf.reduce_all(tf.abs(x_data - x_f) < 1e-6)
    print("Data after IFFT and FFT does not change: {}".format(res))

    # Reshape back to the original input size
    x_after = tf.reshape(x_f, [batch_size, _shape[1], _shape[2]*_shape[3]//2])
    x_after = tf.cast(x_after, tf.complex64)
    res = tf.reduce_all(tf.abs(x_complex - x_after) < 1e-5)
    print("Inputs after the whole process do not change: {}".format(res))

    x_test = tf.reshape(x_after,[-1])
    print(x_test.shape)