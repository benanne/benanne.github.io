---
layout: post
title: The fastest convolutions in Theano with meta-optimization
description: "The fastest convolutions in Theano with meta-optimization"
<!-- modified: 2014-12-09 -->

tags: [deep learning, convolutional neural networks, convnets, Theano, convolution, optimization]

<!-- image:
  feature: abstract-3.jpg
  credit: dargadgetz
  creditlink: http://www.dargadgetz.com/ios-7-abstract-wallpaper-pack-for-iphone-5-and-ipod-touch-retina/ -->

comments: true
share: true
---

<p style='background-color: #ffa; padding: 1.2em;'>
<em>Guest post:</em> <a href="http://ofai.at/~jan.schlueter">Jan Schlüter from the OFAI</a>, a fellow MIR researcher I have met at several conferences, recently added a feature to Theano that fits so well with my <a href="//benanne.github.io/2014/04/03/faster-convolutions-in-theano.html">previous</a> <a href="//benanne.github.io/2014/05/12/fft-convolutions-in-theano.html">two</a> posts on fast convolutions that we decided to include his writeup on my blog. So enjoy the third part of the series, written by Jan!
</p>

Over the past year, [Theano](http://github.com/Theano/Theano) has accumulated several alternative implementations for 2D convolution, the most costly operation in Convolutional Neural Networks.
There is no single implementation that is the fastest for all possible image and kernel shapes,
but with Theano you can mix and match them at will.
Now mixing and matching is something that can be easily automated: Meet meta-optimization!

The idea is to automatically select the fastest available implementation for each individual convolution operation in a Theano function, simply by timing them.
The feature is already available in Theano: If you install the latest version from github, you can activate it by setting the environment variable ``THEANO_FLAGS=optimizer_including=conv_meta,metaopt.verbose=1``.

In the following, I will explain what it does, how it works, and demonstrate that it can outperform all existing convnet libraries.


## Batched convolution

Before we begin, note that the convolution operation in Convolutional Neural Networks (CNNs) as used for Computer Vision is not just a convolution of a single 2D input image with a single 2D filter kernel.
For one, the input image can have multiple channels, such as a color image composed of three values per pixel. It can thus be expressed as a 3D tensor. To match this, the filter kernel has as many values per pixel as the input image, which makes it a 3D tensor as well. When computing the output, each channel is convolved separately with its corresponding kernel, and the resulting images are added up to produce a single 2D output image.
But usually, each convolutional layer returns a multi-channel output (a 3D tensor), which is achieved by learning multiple sets of kernels (a 4D tensor).
Finally, images are often propagated through the network in mini-batches of maybe 64 or 256 items to be processed independently, so the input and output become 4D tensors.

Putting everything together, the batched convolution operation convolves a 4D input tensor with a 4D kernel tensor to produce a 4D output tensor. Obviously, this gives ample of opportunities for parallelization. Add to this the different possible ways of computing a 2D convolution, and you can see why there are so many competing implementations.

*[4D input tensor]: batch size, input channels, input rows, input columns
*[4D kernel tensor]: output channels, input channels, kernel rows, kernel columns
*[4D output tensor]: batch size, output channels, output rows, output columns


## The repertoire

As an actively maintained open-source project with several external contributors, Theano has grown to have access to five convolution implementations:

* a "legacy" implementation that has been created for Theano
* Alex Krizhevsky's **[cuda-convnet](http://code.google.com/p/cuda-convnet)**, via a wrapper already [described by Sander](//benanne.github.io/2014/04/03/faster-convolutions-in-theano.html)
* an **FFT-based convolution** [started by Sander](//benanne.github.io/2014/05/12/fft-convolutions-in-theano.html) and [finished by Arnaud Bergeron](https://github.com/Theano/Theano/pull/1870)
* the **gemm-based convolution** from [Caffe](http://caffe.berkeleyvision.org), [started by Arjun Jain and Frédéric Bastién](https://github.com/Theano/Theano/pull/2002) and [finished by me](https://github.com/Theano/Theano/pull/2033)
* Nvidia's new **[cuDNN](https://developer.nvidia.com/cuDNN) library**, via a wrapper done by [Arnaud](https://github.com/Theano/Theano/pull/2096) and subsequently improved by [Frédéric](https://github.com/Theano/Theano/issues?q=dnn+is%3Aclosed+author%3Anouiz) and [me](https://github.com/Theano/Theano/issues?q=dnn+is%3Aclosed+author%3Af0k)

All of these have their strengths and weaknesses.
cuda-convnet only supports square kernels and places several restrictions on the number of input and output channels and the batch size.
The FFT-based based convolution is applicable to any configuration, but requires a lot of extra memory that practically limits it to small batch and image sizes or very potent graphics cards.
cuDNN requires a GPU of Compute Capability 3.0 or above,
and the convolution ported from Caffe needs some extra memory again.
Finally, the legacy implementation comes free of limitations, but is usually the slowest of the pack.

Depending on the configuration -- that is, the batch size, image shape, filter count and kernel shape --, any of these five implementations can be the fastest.


## Three convolutions per layer

To complicate matters, each convolutional layer in a convnet actually results in three batched convolution operations to be performed in training:

1. The **forward pass**, a valid convolution of images and kernels
2. The **gradient wrt. weights**, a valid convolution of images and the gradient wrt. output
3. The **gradient wrt. input**, a full convolution of the kernels and the gradient wrt. output

For a valid convolution, the kernel is applied wherever it completely overlaps with the input (i.e., it only touches valid data).
For a full convolution, it is applied wherever it overlaps with the input by at least one pixel --
this is equivalent to padding the input with a suitably-sized symmetric border of zeros and applying a valid convolution.

(For the eager ones: The third one in the list above is actually a correlation, because the kernels are not flipped as in the forward pass. And the second one requires the batch size and channels of the input, kernel and output tensors to be swapped. Still all of these can be expressed using the batched convolution operation described in the beginning.)

The "big libraries" (cuda-convnet, Caffe and cuDNN) each come with three algorithms specialized for these three cases, while the FFT-based convolution just distinguishes between valid and full convolutions.


## Cherry-picking

A lot of my work on Theano's convolution was triggered by following Soumith Chintala's [convnet-benchmarks](https://github.com/soumith/convnet-benchmarks) initiative, which set out to compare all freely available Convolutional Neural Network libraries in terms of their performance.
When looking at [some of the first results posted](https://github.com/soumith/convnet-benchmarks/blob/88d4f3b41d86782a8fa1e098c9789c4674bbddb3/README.md), the first thing I noticed was that it would pay off to use a different library for each of the five configurations tested. This has quickly been included as a hypothetical "cherry-picking" row into the result tables.

I took over maintenance of Soumith's Theano benchmark script and evolved it into a handy little tool to compare its convolution implementations for different configurations. Feel free to [download the script](https://github.com/soumith/convnet-benchmarks/tree/master/theano) and follow along.

So let's see what we could gain with cherry-picking in Theano:

~~~
$ SKIP=meta python pylearn2_benchmark.py i3x64x64,k128x7x7,b64
Using gpu device 0: GeForce GTX 780 Ti

CONFIG: input = 3 x 64 x 64 * ker = 3 x 128 x 7 x 7 ( bs = 64 , stride = 1 )
theano.tensor.nnet.conv.conv2d                     ==> fprop         ==>      43
theano.tensor.nnet.conv.conv2d                     ==> bprop inputs  ==>      44
theano.tensor.nnet.conv.conv2d                     ==> bprop weights ==>     185

theano.sandbox.cuda.fftconv.conv2d_fft             ==> fprop         ==>      19
theano.sandbox.cuda.fftconv.conv2d_fft             ==> bprop inputs  ==>      26
theano.sandbox.cuda.fftconv.conv2d_fft             ==> bprop weights ==>      20

(auto) theano.sandbox.cuda.dnn.GpuDnnConv          ==> fprop         ==>       4
(auto) theano.sandbox.cuda.dnn.GpuDnnConv          ==> bprop inputs  ==>       7
(auto) theano.sandbox.cuda.dnn.GpuDnnConv          ==> bprop weights ==>       6

(auto) theano.sandbox.cuda.blas.GpuCorrMM          ==> fprop         ==>       6
(auto) theano.sandbox.cuda.blas.GpuCorrMM          ==> bprop inputs  ==>       7
(auto) theano.sandbox.cuda.blas.GpuCorrMM          ==> bprop weights ==>      10

pylearn2.sandbox.cuda_convnet(partial_sum=None)    ==> fprop         ==>       7
pylearn2.sandbox.cuda_convnet(partial_sum=None)    ==> bprop inputs  ==>      11
pylearn2.sandbox.cuda_convnet(partial_sum=None)    ==> bprop weights ==>      47

pylearn2.sandbox.cuda_convnet(partial_sum=1)       ==> fprop         ==>       7
pylearn2.sandbox.cuda_convnet(partial_sum=1)       ==> bprop inputs  ==>      11
pylearn2.sandbox.cuda_convnet(partial_sum=1)       ==> bprop weights ==>      13
~~~
What we see here are the respective computation times in milliseconds for a particular configuration (tensor shapes) for the legacy implementation, FFT-based convolution, cuDNN, gemm-based convolution and cuda-convnet (with two different values for a tuning parameter).
For this layer, cuDNN would be the optimal choice.

Let's try a second configuration:

~~~
$ SKIP=meta python pylearn2_benchmark.py i32x15x80,k64x5x5,b256
Using gpu device 0: GeForce GTX 780 Ti

CONFIG: input = 32 x 15 x 80 * ker = 32 x 64 x 5 x 5 ( bs = 256 , stride = 1 )
theano.tensor.nnet.conv.conv2d                     ==> fprop         ==>     146
theano.tensor.nnet.conv.conv2d                     ==> bprop inputs  ==>     182
theano.tensor.nnet.conv.conv2d                     ==> bprop weights ==>     162

theano.sandbox.cuda.fftconv.conv2d_fft             ==> fprop         ==>      20
theano.sandbox.cuda.fftconv.conv2d_fft             ==> bprop inputs  ==>      24
theano.sandbox.cuda.fftconv.conv2d_fft             ==> bprop weights ==>      15

(auto) theano.sandbox.cuda.dnn.GpuDnnConv          ==> fprop         ==>      18
(auto) theano.sandbox.cuda.dnn.GpuDnnConv          ==> bprop inputs  ==>      23
(auto) theano.sandbox.cuda.dnn.GpuDnnConv          ==> bprop weights ==>      25

(auto) theano.sandbox.cuda.blas.GpuCorrMM          ==> fprop         ==>      22
(auto) theano.sandbox.cuda.blas.GpuCorrMM          ==> bprop inputs  ==>      29
(auto) theano.sandbox.cuda.blas.GpuCorrMM          ==> bprop weights ==>      30

pylearn2.sandbox.cuda_convnet(partial_sum=None)    ==> fprop         ==>      16
pylearn2.sandbox.cuda_convnet(partial_sum=None)    ==> bprop inputs  ==>      20
pylearn2.sandbox.cuda_convnet(partial_sum=None)    ==> bprop weights ==>      40

pylearn2.sandbox.cuda_convnet(partial_sum=1)       ==> fprop         ==>      16
pylearn2.sandbox.cuda_convnet(partial_sum=1)       ==> bprop inputs  ==>      21
pylearn2.sandbox.cuda_convnet(partial_sum=1)       ==> bprop weights ==>      28
~~~
This time, the FFT-based convolution is faster, but the truly optimal choice would be combining it with cuda-convnet.

We see that the meta-optimizer should not just cherry-pick a different implementation per convolutional layer, but even a different implementation for each of the three convolutions in a layer -- something that was not possible in Theano before (nor in any other library I am aware of).


## The "swapping trick"

As you recall, cuda-convnet, Caffe and cuDNN come with specialized algorithms for the three convolutions per layer.
Interestingly, when porting the gemm-based convolution from Caffe to Theano, I noticed that the effort I put in properly using its two backward pass algorithms when applicable did not always pay off: For some configurations, it was faster to just use the forward pass algorithm instead, transposing tensors as needed.
I thus added [a shape-based heuristic](https://github.com/Theano/Theano/blob/1477ded8740636c381076b8720055d6c2be64590/theano/sandbox/cuda/opt.py#L1372-1400) to select the fastest algorithm for the gemm-based convolution (making Theano's port faster than Caffe for some configurations).

When adding support for Nvidia's cuDNN library, Arnaud understandably assumed that it would hide this complexity from the user and select the optimal algorithm internally. So at first, Theano did not tell cuDNN whether a particular convolution's purpose was a forward pass or one of the backward passes. When I [changed the implementation](https://github.com/Theano/Theano/pull/2273) accordingly, I again noticed that while performance generally improved a lot, for some configurations, using the "wrong" algorithm was actually faster.

Just as for Caffe, we can use this knowledge to be faster than cuDNN.
As the implementation is unknown, we cannot easily define a heuristic for choosing between the cuDNN algorithms.
However, the meta-optimizer can just try all applicable algorithms and see which one is the fastest.
I found it to suffice to just try two algorithms per convolution:

* For the forward pass, try the "correct" algorithm and the gradient wrt. weights (both are valid convolutions)
* For the gradient wrt. weights, try the "correct" algorithm and the forward pass
* For the gradient wrt. inputs, try the "correct" algorithm and the forward pass (with additional zero padding to make it a full convolution)

I call this the "swapping trick" because it often leads to the first two algorithms being swapped.


## Implementation

To understand why Theano was a perfect fit to add automatic algorithm selection, we will need to explain a bit of its inner workings.

First, Theano is not a neural network library, but a mathematical expression compiler.
In contrast to, say, Caffe, its basic components are not neural network layers, but mathematical operations.
Implementing a neural network is done by composing the expression for the forward pass (which will probably include matrix multiplications, vector additions, elementwise nonlinearities and possibly batched convolution and pooling), using this to build an expression for the training cost, and then letting Theano transform it into expressions for the gradients wrt. the parameters to be learned.
Finally, the expressions are compiled into functions that evaluate them for specific settings of the free variables (such as a mini-batch of training data).

But right before an expression is compiled, it is *optimized*, and this is where all the magic happens.
The expression is represented as a graph of Apply nodes (operations) and Variable nodes (the inputs and outputs of an operation), and Theano comes with a bunch of *graph optimizers* that modify the graph to produce the same result either more efficiently or more numerically stable.
\\
One particular graph optimizer moves convolution operations from the CPU to the GPU by replacing the respective Apply node and adding the necessary transfer operations around it.
A whole set of graph optimizers then replaces the legacy GPU convolution operation with one of the more efficient implementations available in Theano. These optimizers have relative priorities and can be enabled and disabled by the user.

The new meta-optimizer is just another graph optimizer with a twist: When it encounters a convolution operation, it applies each of the set of available graph optimizers (plus the cuDNN "swapping trick" optimizer) in sequence, each time compiling and executing the subgraph performing the convolution, and chooses the one resulting in the best performance.
(Finally, this explains why it's called *meta*-optimization.)
\\
As the basic components in Theano are the mathematical operations, there is no extra work needed to be able to choose different implementations for the three convolutions per layer: All Theano sees when optimizing and compiling an expression is a graph containing several anonymous convolution operations, so it will naturally optimize each of them separately.


## Practical gains

Let us now put the meta-optimizer to test using the benchmark script mentioned in the cherry-picking section:

~~~
$ THEANO_FLAGS=metaopt.verbose=1 SKIP=legacy,gemm,fft,convnet,dnn python pylearn2_benchmark.py i128x36x12,k64x6x3,b256
Using gpu device 0: GeForce GTX 780 Ti

CONFIG: input = 128 x 36 x 12 * ker = 128 x 64 x 6 x 3 ( bs = 256 , stride = 1 )
ConvMetaOptimizer meta-optimizing GpuConv{valid, (1, 1), None, (3, 6), True, (128, 12, 36), (3, 6)}(GpuFromHost.0, GpuFromHost.0) (5 choices):
* local_conv_fft_full: not applicable
* local_conv_fft_valid: 0.012958 sec
* local_conv_dnn: 0.021169 sec
* local_conv_gemm: 0.03973 sec
* local_conv_dnn_alternative: 0.044379 sec
= local_conv_fft_valid
(experimental) meta-optimizer                      ==> fprop         ==>      12
ConvMetaOptimizer meta-optimizing GpuConv{full, (1, 1), None, (3, 6), True, (64, 10, 31), (3, 6)}(GpuFromHost.0, GpuFromHost.0) (5 choices):
* local_conv_fft_full: 0.019099 sec
* local_conv_fft_valid: not applicable
* local_conv_dnn: 0.032979 sec
* local_conv_gemm: 0.028478 sec
* local_conv_dnn_alternative: 0.015099 sec
= local_conv_dnn_alternative
(experimental) meta-optimizer                      ==> bprop inputs  ==>      15
ConvMetaOptimizer meta-optimizing GpuConv{valid, (1, 1), None, (10, 31), False, (256, 12, 36), (10, 31)}(GpuFromHost.0, GpuFromHost.0) (5 choices):
* local_conv_fft_full: not applicable
* local_conv_fft_valid: 0.011441 sec
* local_conv_dnn: 0.030338 sec
* local_conv_gemm: 0.025984 sec
* local_conv_dnn_alternative: 0.031552 sec
= local_conv_fft_valid
(experimental) meta-optimizer                      ==> bprop weights ==>      12
~~~

In verbose mode, the meta-optimizer reports which implementations are tested, how each of them performs and which one is finally chosen.
For the configuration at hands, it turns out that the FFT-based implementation is fastest for the forward pass and the gradient wrt. weights, and cuDNN is fastest for the gradient wrt. inputs -- but only when using the "wrong" algorithm for it (namely, cuDNN's forward pass algorithm with zero padding, tried according to the swapping trick).
In all three instances, the optimal algorithm is about twice as fast as just choosing cuDNN, which would have been Theano's current default behavior.

When training a full network, the impact will generally be smaller, because the convolution operations only constitute a part of the expressions evaluated (but often the most costly part).
The improvement also heavily depends on the input and kernel shapes -- for a wide range of configurations, just using cuDNN for all convolutions is nearly optimal.
Still, a colleague of Sander reported a threefold performance improvement for a network trained for a Kaggle competition, with the meta-optimizer combining FFT, Caffe, and cuDNN with and without the swapping trick.
<!-- For the curious: the configurations were `i1x104x104,k128x9x9,b8` and `i128x96x96,k16x1x1,b8`. -->

To get an estimate on how much Theano could help for your use case, just run [the benchmark script](https://github.com/soumith/convnet-benchmarks/tree/master/theano) for the configurations occurring in a forward pass through your network.
If you already use Theano, just set `THEANO_FLAGS=optimizer_including=conv_meta` to rest assured you will always make the most out of the time (and electricity!) you spend on training your networks.


## Future

While the basic machinery is in place and works fine, there are a lot of conceivable improvements:

* The meta-optimizer should cache its results on disk to speed up repeated compilations of the same graph.
* Right now, the meta-optimizer uses all available convolution operations in Theano; it should be possible to control this.
* As cuda-convnet is not included in Theano, but an external project (Pylearn2), it is not included in the meta-optimizer. However, it is possible to register additional optimizers at runtime via `theano.sandbox.cuda.opt.conv_metaopt.register()`. It would be nice to write such a pluggable optimizer for cuda-convnet.
* Similarly, it would be nice to have a wrapper for cuda-convnet2 (in a separate repository) along with an optimizer to be registered with the meta-optimizer.
* Currently, meta-optimization can only be used for non-strided valid or full convolutions, because this is what the legacy implementation is limited to. Changing this would require [some refactoring](https://github.com/Theano/Theano/issues/2268#issuecomment-63621626), but lead to cleaner code and slightly improved performance.
* Finally, it could be worthwhile to repeat the same for the pooling operation of CNNs: Port additional implementations to Theano, benchmark them and add a meta-optimizer.

Watch [Issue #2072](https://github.com/Theano/Theano/issues/2072) on github for any progress on this, or even better, step in and implement one of these features if you can use it!
Both that issue and [theano-dev](https://groups.google.com/forum/#!forum/theano-dev) are well-suited to ask for hints about implementing any of these TODOs -- we'd be glad to have you on board.
