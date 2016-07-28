# Purpose
This repository is an implementation of the paper [Spectral Representations for Convolutional Neural Networks](http://arxiv.org/abs/1506.03767) in [Tensorflow](http://tensorflow.org/)

# Install
### Installing Tensorflow with GPU support:
```bash
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.9.0rc0-cp27-none-linux_x86_64.whl
sudo pip install --upgrade $TF_BINARY_URL
```

### Installing python packages:
It's best to install [Anaconda](https://www.continuum.io/downloads) which comes with numpy
```bash
pip install tqdm
conda install opencv
```

# Status
Currently, the spectral parametrization is resulting in filters that are inferior to the spatial parametrization and spectral pooling has not been implemented

# Reasons why it might be failing

- Enforcing conjugate symetry is done via a slow method of creating individual variables for each of the free parameters in `spectral_to_variable()`. Currently 
this is turned off (look for `Option 2` in the code) because it didn't seem to improve results dramatically and is very slow. I may be implementing this incorrectly.
- Tensorflow may have a weirdness [in the way they implement the gradient for ifft in `_BatchIFFTGrad()`](https://github.com/tensorflow/tensorflow/blob/73ced9d797056c7e67a06ed2098dd809d85ec44a/tensorflow/python/ops/math_grad.py)
- I am not directly accounting for 

# Findings and other tests

L1 Regularizing in the spectral domain results in sparse spectral representations of spatial features (as expected) and significantly cleans up the spatial representations.

---

Initializing spectral filters to be the final filter learned by the spatial parameterization is stable.
To confirm this, the following code in the main loop should be replaced:
```
fft = FFTConvTest(operations='fft')
```
with:
```python
fft = FFTConvTest(operations='fft', initialization={'conv1': baseline.spectral_conv1.eval(session=baseline.sess),
                                                           'conv2': baseline.spectral_conv2.eval(session=baseline.sess)})
```
---
Tensorflow's FFT and IFFT gradients are inverses of one another.
To confirm this, the following code should be added to `conv2d()`
```python
spatial_filter_fft = tf.real(tf.batch_ifft2d(tf.batch_fft2d(tf.complex(spatial_filter_for_fft, spatial_filter_for_fft * 0.0))))
spatial_filter = tf.transpose(spatial_filter_fft, [2, 3, 0, 1])
```
---
Tensorflow's FFT and IFFT  operations are equivalent to numpy. 
To confirm this, the following code should be added to `train()`.
```python
pixel = spatial_filters[0, 0].real
freq = spectral_filters[0, 0]
pixel_to_freq = np.fft.fft2(pixel).real
freq_to_pixel = np.fft.ifft2(freq).real
if np.abs(np.max(pixel) - np.max(freq_to_pixel)) > 1e-5 and np.abs(np.min(freq.real) - np.max(pixel_to_freq)) > 1e-5:
    print 'spatial(min:{} max:{})  spectral_to_spatial(min:{} max:{}) '.format(np.min(pixel),
                                                                               np.max(pixel),
                                                                               np.min(freq_to_pixel),
                                                                               np.max(freq_to_pixel))
    print 'spectral(min:{} max:{}) spatial_to_spectral(min:{} max:{})'.format(np.min(freq.real),
                                                                              np.max(freq.real),
                                                                              np.min(pixel_to_freq),
                                                                              np.max(pixel_to_freq))
```
---
3d FFTs do not seem to help. Even though normal `tf.nn.conv2d()` convolutions appear to be 3d (height, width, channel)
applying `tf.batch_fft3d()` instead of the `tf.batch_fft2d` does not learn valuable filters, although the filters are sparse and result in structured spatial filters.
To confirm this, the following code should be added to `FFTConvTest` or replace `FFTConvTest.fft_conv()` entirely
```
def fft3d_conv(self, source, filters, width, height, stride, activation='relu', name='fft_conv'):
        channels = source.get_shape().as_list()[3]

    with tf.variable_scope(name):
        init = self.random_spatial_to_spectral(filters, channels, height, width)

        if name in self.initialization:
            init = self.initialization[name]

        # w_real = tf.Variable(init.real, dtype=tf.float32, name='real')
        # w_imag = tf.Variable(init.imag, dtype=tf.float32, name='imag')
        # w = tf.cast(tf.complex(w_real, w_imag), tf.complex64)

        w = self.spectral_to_variable(init)
        b = tf.Variable(tf.constant(0.1, shape=[filters]))

    # Transform the spectral parameters into a spatial filter
    # and reshape for tf.nn.conv2d

    complex_spatial_filter = tf.batch_ifft3d(w)
    spatial_filter = tf.real(complex_spatial_filter)
    spatial_filter = tf.transpose(spatial_filter, [2, 3, 1, 0])

    w = tf.transpose(w, [1, 0, 2, 3])

    conv = tf.nn.conv2d(source, spatial_filter, strides=[1, stride, stride, 1], padding='SAME')
    output = tf.nn.bias_add(conv, b)
    output = tf.nn.relu(output) if activation is 'relu' else output

    return output, spatial_filter, w
    
```
---
Parameterizing a pure FFT implementation of convolution (that is, doing point-wise multiplication in the spectral domain) in
the spatial domain results in sparse features, but does not produce filters comparable to normal spatial convolution.
To confirm this, uncomment `Option 3` in `FFTConvTest.fft_conv_pure()`