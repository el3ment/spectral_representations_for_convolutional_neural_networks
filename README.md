
Tensorflow's FFT and IFFT gradients are inverses of one another.
To confirm this, the following code was added to `conv2d()`
```
    spatial_filter_fft = tf.real(tf.batch_ifft2d(tf.batch_fft2d(tf.complex(spatial_filter_for_fft, spatial_filter_for_fft * 0.0))))
    spatial_filter = tf.transpose(spatial_filter_fft, [2, 3, 0, 1])
```

Tensorflow's FFT and IFFT  operations are equivalent to numpy. 
To confirm this, the following code was added to `train()`.
```
    pixel = spatial_filters[0, 0].real
    freq = spectral_filters[0, 0]
    pixel_to_freq = np.fft.fft2(pixel).real
    freq_to_pixel = np.fft.ifft2(freq).real
    if np.abs(np.max(pixel) - np.max(freq_to_pixel)) > 1e-5 and \
                    np.abs(np.min(freq.real) - np.max(pixel_to_freq)) > 1e-5:
        print 'spatial(min:{} max:{})  spectral_to_spatial(min:{} max:{}) '.format(np.min(pixel),
                                                                                   np.max(pixel),
                                                                                   np.min(freq_to_pixel),
                                                                                   np.max(freq_to_pixel))
        print 'spectral(min:{} max:{}) spatial_to_spectral(min:{} max:{})'.format(np.min(freq.real),
                                                                                  np.max(freq.real),
                                                                                  np.min(pixel_to_freq),
                                                                                  np.max(pixel_to_freq))
```