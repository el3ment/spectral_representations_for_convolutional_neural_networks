# Purpose
This repository is an implementation of the paper [Spectral Representations for Convolutional Neural Networks](http://arxiv.org/abs/1506.03767) in [Tensorflow](http://tensorflow.org/)

# Next Steps
Spectral parametrization now results in accuracies that are comparable to the baseline! 

Currently, the way we are enforcing conjugate symmetry is via two manually constructed tensors that each contain half free and half dependent parameters. 
Unfortunately this is extremely slow, but it is accurate. The optimal solution to this slowness problem is to use real-valued FFT and iFFT functions so that steps
in the direction of the gradient will always result in valid spectral representations. Implementing these real-valued FFT functions will be much faster because they only
require half as many FFT operations, *and* there is no need to call the `RecoverMap`, `RemoveRedundency`, and `TreatCornerCases` functions which each require additional loops through
the tensor. 

The next step then is to implement real-valued FFT functions in Tensorflow. To do this, we will need to [add a new tensorflow op](https://www.tensorflow.org/versions/r0.9/how_tos/adding_an_op/index.html) using the 
[cufftExecR2C() and cufftExecC2R()](http://docs.nvidia.com/cuda/cufft/index.html#fft-types) functions from cuda. This likely involves simply adding a new op to [fft_ops.cc](https://github.com/tensorflow/tensorflow/blob/d42facc3cc9611f0c9722c81551a7404a0bd3f6b/tensorflow/core/kernels/fft_ops.cc)
that uses a slightly modified `DoFFT()` function that calls R2C and C2R instead.
 
In addition, spectral pooling remains to be implemented, and a faster [implementation of fftshift](https://devtalk.nvidia.com/default/topic/515723/does-cuda-provide-fftshift-function-like-matlab-/) leveraged (or apply the padding directly to the unshifted vector)


# Theory / Questions
- What non-linear functions maintain conjugate symmetry (which would enable a fully spectral network)?
- What is the proper way to handle the channel dimension when doing a pure spectral convolution?
- Is there a need to handle the difference between convolution and cross-correlation?
- How should we scale the spectral regularization parameter with filter height/width?
- How does the regularization of a spectrally parameterized filter relate to the regularization of a spatially parameterized filter?
- How does the gradient of a spectrally parameterized filter relate to the regularization of a spatially parameterized filter, and how does this impact learning rate?


# Install
### Installing Tensorflow with GPU support:
```bash
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.9.0rc0-cp27-none-linux_x86_64.whl
sudo pip install --upgrade $TF_BINARY_URL
```

### Installing python:
It's best to install [Anaconda](https://www.continuum.io/downloads) which comes complete with numpy and conda (used for opencv install)

### Installing needed python packages
```bash
pip install tqdm
conda install opencv
```

# Findings and other tests

L1 Regularizing in the spectral domain results in sparse spectral representations of spatial features (as expected) and significantly cleans up the spatial representations.

---

Initializing spectral filters to be the final filter learned by the spatial parametrization is stable.
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