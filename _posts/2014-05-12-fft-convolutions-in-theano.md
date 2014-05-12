---
layout: post
title: Even faster convolutions in Theano using FFTs
description: "Even faster convolutions in Theano using Fast Fourier Transforms"
<!-- modified: 2014-03-29 -->

tags: [deep learning, convolutional neural networks, convnets, Theano, convolution, FFT, fast Fourier transform]

<!-- image:
  feature: abstract-3.jpg
  credit: dargadgetz
  creditlink: http://www.dargadgetz.com/ios-7-abstract-wallpaper-pack-for-iphone-5-and-ipod-touch-retina/ -->

comments: true
share: true
---

[Last month](http://benanne.github.io/2014/04/03/faster-convolutions-in-theano.html) I wrote about how you can use the cuda-convnet wrappers in pylearn2 to get up to 3x faster GPU convolutions in Theano. Since then I've been working on an FFT-based convolution implementation for Theano. Preliminary tests indicate that this approach is again 2-4x faster than the cuda-convnet wrappers.

I wrote the code in pure Python, using [scikits.cuda](https://github.com/lebedov/scikits.cuda) and [PyCUDA](http://mathema.tician.de/software/pycuda/) to do the heavy lifting. The Theano team is [currently working on integrating this code into Theano](https://groups.google.com/forum/#!topic/theano-users/6xiFFpBBDq0). They also plan to create a proper C/CUDA implementation to guarantee the best performance.

I put everything up on GitHub, you can find the code there, or clone it and try it yourself:

* **[https://github.com/benanne/theano_fftconv](https://github.com/benanne/theano_fftconv)**

## FFT-based convolution

The Fourier transform of a convolution of two functions is the product of the Fourier transforms of those functions. This is the [convolution theorem](http://en.wikipedia.org/wiki/Convolution_theorem). This result can be used to quickly compute convolutions in the Fourier domain, since an elementwise product is much less computationally intensive than a convolution.

However, there is a price to be paid: the inputs need to be transformed using the Fast Fourier Transform (FFT), and the product of these transformed inputs needs to be transformed again using the inverse FFT. Depending on the sizes of the inputs, these costs can be pretty significant, so sometimes it is a better idea to just compute the convolution in the original domain.

I was somewhat surprised to learn that all popular implementations of convolutional neural networks (CNNs) use the latter approach, including that of Theano and cuda-convnet. The reason is that typically, convolutions in CNNs involve relatively small filters, so I think people just assumed it wasn't worth it.

However, a paper published at ICLR 2014 recently caught my eye: [Fast Training of Convolutional Networks through FFTs](http://openreview.net/document/aa6ab717-ca19-47e1-a958-823b9a106ca9#aa6ab717-ca19-47e1-a958-823b9a106ca9) by Mathieu, Henaff and LeCun. They implemented the FFT-based approach in the [Torch7 framework](http://torch.ch/) and compared its performance to Torch7's own 'classical' implementation. They concluded that it is actually advantageous to use FFT-based convolutions in CNNs in many cases. 

The reason is actually quite straightforward: compared to the general case, the overhead of computing the FFTs of the inputs is drastically reduced. We need to compute the convolution of each input example in a given minibatch with each filter. If there are `m` examples in the minibatch with `k` input channels, and `n` filters, this means we need to compute `m * n * k` convolutions. In the Fourier domain, this turns into `m * n * k` elementwise products. However, **we only need to compute the FFT of each input example and each filter once**. So the total number of FFTs to compute is not `2 * m * n * k`, but `(m + n) * k`.

But that's not everything: the output of a convolutional layer in a CNN is actually a sum of convolutions across all `k` input channels. Because the FFT is a linear operator, we can compute this sum in the Fourier domain, and then take the IFFT of this sum (instead of the other way around). This means we only need to compute `m * n` IFFTs, instead of `m * n * k`. It turns out that these savings can be very significant.

## A CUDA/C-less Theano implementation

So this got me thinking that it should be possible to do the same thing in Theano. Theano already intelligently replaces convolution operators in computational graphs with their GPU-based counterparts in the optimization phase. If an FFT-based implementation was added, it could do the same with that version instead.

I set out to implement this, but unfortunately my knowledge of CUDA is nonexistent, and my knowledge of C can be called rusty at best. So I sought to avoid both. Enter [scikits.cuda](https://github.com/lebedov/scikits.cuda), which offers all the necessary primitives: forward and inverse FFTs, and complex products (the FFT of a real signal is complex and symmetric).

Luckily, scikits.cuda is built on top of [PyCUDA](http://mathema.tician.de/software/pycuda/), and the Theano docs have some examples of how to implement PyCUDA-based operators. Essentially I just had to glue everything together.

## Implementation details

As mentioned earlier, an FFT-based convolution can be broken up into 3 parts: an FFT of the input images and the filters, a bunch of elementwise products followed by a sum across input channels, and then an IFFT of the outputs. I decided to implement each of these as a separate Theano operator. That way, the optimizer could detect if the same inputs or filters are used in multiple convolutions, and only compute them once. At the moment I'm still unsure whether this is beneficial - perhaps some additional performance could be gained by combining everything into a single, monolithic FFT-convolution operator. But that's a discussion for another time.

The FFT and IFFT operators were the easiest. scikits.cuda exposes a nice API to perform **batched FFTs**. This allows for GPU-parallelism to be exploited when many FFTs of the same size have to be computed. This is precisely our use case. The API uses the cuFFT implementation internally, which is a part of CUDA.

Interestingly, the authors of the paper I mentioned earlier claim that using cuFFT is not an option because it does not allow to exploit this type of parallelism, so they made their own CUDA FFT implementation instead. However, I got pretty good results using cuFFT, so I don't know what lead them to make this claim. Perhaps the batched FFT is a recent addition to cuFFT. The same batched approach can be used for the IFFT.

The tough part was performing the actual convolution in the Fourier domain, by computing the complex elementwise products and summing across the input channels. Theano does not have support for complex numbers, so some trickery was required to convert complex arrays into real arrays with an extra trailing dimension of size 2, to contain the real and imaginary parts of the numbers.

I tried a number of different approaches, but what worked best in the end is interpreting the operation as a dot product. A dot product is precisely that: an elementwise product with some broadcasting, followed by summing out a particular dimension. So by reshaping the Fourier-transformed inputs and filters, the multiply-and-sum operation could be translated into a set of dot products. This is great, because GPUs are really good at computing dot products quickly.

It turns out that recent versions of cuBLAS also support **batched dot products**, which offer the same performance advantages as batched FFTs. Since we need to perform a large number of dot products with the same shapes, this was again a perfect match for our use case. The particular function I needed to compute a batched complex-valued dot product is `cublasCgemmBatched`. Unfortunately this wasn't available through scikits.cuda yet, but it wasn't hard to add the necessary wrappers. I sent a [pull request](https://github.com/lebedov/scikits.cuda/pull/52) and it is now included (so make sure to get the latest version of scikits.cuda from git if you want to try this).

## Proof of concept

So far I've only implemented the *valid* convolution. Using the implementation in the context of a CNN will also require support for full convolutions - but this is easy to mimic by padding the input with zeros. I have not implemented an optimization that swaps out Theano's own convolution operator with the FFT-based version, but that is something the Theano team is currently working on.

Preliminary benchmarks show that this implementation is typically faster than cuda-convnet. The table below shows the duration of a single valid convolution computation with the given input and filter shapes, measured on a GeForce GTX 680, averaged across 10 runs, and not taking into account the warmup that the FFT-based implementation requires (the first run will be a bit slower because the FFT plans need to be created).

Following Theano conventions, the input shape is given as `(batch size, number of input channels, width, height)` and the filter shape is given as `(number of filters, number of input channels, width, height)`. Durations are given for Theano's own `conv2d` implementation, the cuda-convnet wrappers from pylearn2, and the FFT-based implementation. The speedup of the FFT-based implementation over the cuda-convnet wrappers is also given.

|input shape|filter shape|Theano's own|cuda-convnet|FFT-based|speedup|
|:---------:|:----------:|:----:|:----------:|:-:|:-:|
|(64, 3, 96, 96)|(128, 3, 16, 16)|388.9 ms|156.9 ms|117.3 ms|1.34x|
|(64, 128, 32, 32)|(64, 128, 8, 8)|233.9 ms|87.4 ms| 27.1 ms|3.23x|
|(128, 32, 54, 54)|(64, 32, 6, 6)|457.5 ms|107.6 ms|52.2 ms|2.06x|
|(128, 128, 16, 16)|(128, 128, 8, 8)|133.4 ms| 43.5 ms| 18.6 ms|2.34x|
|(128, 1024, 32, 32)|(128, 1024, 4, 4)|6246.2 ms|1283.5 ms|357.8 ms|3.59x|

In all cases we get a nice speedup. This approach seems to be the most beneficial when the number of input channels is large - this makes sense, as this is the dimension that is summed over in the batched dot product. But even when this number is small (e.g. 3) it's still faster.

## Try it out

As mentioned in the introduction, you can grab the code for this at:

* **[https://github.com/benanne/theano_fftconv](https://github.com/benanne/theano_fftconv)**

All the relevant code is in the file [`fftconv.py`](https://github.com/benanne/theano_fftconv/blob/master/fftconv.py). The file [`cufftop.py`](https://github.com/benanne/theano_fftconv/blob/master/cufftop.py) was mainly used for experimentation, and contains some alternative implementations of the multiply-and-sum step.

Note that the latest revision of scikits.cuda is required, to ensure that the `cublasCgemmBatched` function is available. You'll also need a working installation of PyCUDA, as this is a dependency of scikits.cuda. And of course, you'll need Theano and a working CUDA installation.

If you're patient, you can also wait until the code is available in Theano. Chances are you'll be able to use it without modifying your existing code, as they are also building an optimization that will replace Theano's own convolutions with the FFT-based implementation. And if you're very patient, you can wait until they build the CUDA/C version, which will eliminate the scikits.cuda and PyCUDA dependencies, and hopefully it will be a bit faster as well due to the reduced overhead.

The code to compute the numbers in the table above is in the file [`speedtest.py`](https://github.com/benanne/theano_fftconv/blob/master/speedtest.py). This script also checks whether the output of all three implementations is the same (up to a given tolerance). More numbers for different input/filter shapes and different GPUs are welcome, so if you run this script on your own machine(s), feel free to send me the results.

Feedback is welcome, and if you'd like to help with integrating this into Theano, [join the conversation at the theano-users group](https://groups.google.com/forum/#!topic/theano-users/6xiFFpBBDq0)!
