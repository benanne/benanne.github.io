---
layout: post
title: "New Lasagne feature: arbitrary expressions as layer parameters"
description: "Lasagne now supports arbitrary Theano expressions as layer parameters, creating more flexibility and allowing easier code reuse."
<!-- modified: 2014-12-09 -->

tags: [deep learning, neural networks, Theano, Lasagne]

image:
  feature: lasagna.jpg
comments: true
share: true
---

<p style='background-color: #ffa; padding: 1.2em;'>
This post is another collaboration with <a href="http://ofai.at/~jan.schlueter">Jan Schlüter from the OFAI</a> (<a href="https://github.com/f0k">@f0k</a> on GitHub), a fellow MIR researcher and one of the lead developers of <a href="http://lasagne.readthedocs.org/">Lasagne</a>. He recently added a cool new feature that we wanted to highlight: enabling the use of arbitrary Theano expressions as layer parameters.
</p>

As many of you probably know, Jan Schlüter and I are part of the team that develops [Lasagne](http://lasagne.readthedocs.org/), a lightweight neural network library built on top of [Theano](http://deeplearning.net/software/theano/).

One of the key [design principles](http://lasagne.readthedocs.org/en/latest/user/development.html#philosophy) of Lasagne is *transparency*: we try not to hide Theano or numpy behind an additional layer of abstractions and encapsulation, but rather expose their functionality and data types and try to follow their conventions. This makes it very easy to learn how to use Lasagne if you already know how to use Theano -- there just isn't all that much extra to learn. But most importantly, it allows you to easily mix and match parts of Lasagne with vanilla Theano code. This is the way Lasagne is meant to be used.

In keeping with this philosophy, Jan recently added a feature that we've been discussing early on in designing the API ([#11](https://github.com/Lasagne/Lasagne/issues/11)): it allows any learnable layer parameter to be specified as a mathematical expression evaluating to a correctly-shaped tensor. Previously, layer parameters had to be Theano shared variables, i.e., naked tensors to be learned directly. **This new feature makes it possible to constrain network parameters in various, potentially creative ways.** Below, we'll go through a few examples of what is now possible that wasn't before.

## Default case

Let's create a simple fully-connected layer of 500 units on top of an input layer of 784 units.

{% highlight python %}
from lasagne.layers import InputLayer, DenseLayer
batch_size = 64
l1 = InputLayer((batch_size, 784))
l2 = DenseLayer(l1, num_units=500)
{% endhighlight %}

## Autoencoder with tied weights

Autoencoders with tied weights are a common use case, and until now implementing them in Lasagne was a bit tricky. Weight sharing in Lasagne has always been easy and intuitive:

{% highlight python %}
l2 = DenseLayer(l1, num_units=500)
l3 = DenseLayer(l1, num_units=500, W=l2.W)
# l2 and l3 now share the same weight matrix!
{% endhighlight %}

... but in an autoencoder, you want the weights of the decoding layer to be the *transpose* of the weights of the encoding layer. So you would do:

{% highlight python %}
l2 = DenseLayer(l1, num_units=500)
l3 = DenseLayer(l2, num_units=784, W=l2.W.T)
{% endhighlight %}

... but that didn't work before: `l2.W.T` is a Theano expression, but not a Theano shared variable as was expected. This is counter-intuitive, and indeed, [people expected it to work](https://groups.google.com/forum/#!searchin/lasagne-users/tied$20weights/lasagne-users/ky78GBSgnBI/z10Br4p4kHMJ) and were disappointed to find out that it didn't. With the new feature this is no longer true. The above will work just fine. Yay!

## Factorized weights

To reduce the number of parameters in your network (e.g. to prevent overfitting), you could force large parameter matrices to be *low-rank* by factorizing them. In our example from before, we could factorize the 784x500 weight matrix into the product of a 784x100 and a 100x500 matrix. The number of weights of the layer then goes down from 392000 to 128400 (not including the biases).

{% highlight python %}
import theano
import theano.tensor as T
from lasagne.init import GlorotUniform
from lasagne.utils import floatX
w_init = GlorotUniform()
w1 = theano.shared(floatX(w_init((784, 100))))
w2 = theano.shared(floatX(w_init((100, 500))))
l2 = DenseLayer(l1, num_units=500, W=T.dot(w1, w2))
{% endhighlight %}

Granted, this was possible before by inserting a biasless linear layer:
{% highlight python %}
l2_a = DenseLayer(l1, num_units=100, b=None, nonlinearity=None)
l2 = DenseLayer(l2_a, num_units=500)
{% endhighlight %}

Other types of factorizations [may also be worth investigating!](http://arxiv.org/abs/1509.06569)

## Positive weights

If you want to force the weights of a layer to be positive, you can learn their logarithm:

{% highlight python %}
from lasagne.init import Normal
w = theano.shared(floatX(Normal(0.01, mean=-10)((784, 500))))
l2 = DenseLayer(l1, num_units=500, W=T.exp(w))
{% endhighlight %}

You could also use `T.softplus(w)` instead of `T.exp(w)`. You might also be tempted to try sticking a ReLU in there (`T.maximum(w, 0)`), but note that applying the linear rectifier to the weight matrix would lead to many of the underlying weights getting stuck at negative values, as the linear rectifier has zero gradient for negative inputs!

## Positive semi-definite weights

There are plenty of other creative uses, such as constraining weights to be positive semi-definite (for whatever reason):

{% highlight python %}
l2 = DenseLayer(l1, num_units=500)
w = theano.shared(floatX(w_init((500, 500))))
w_psd = T.dot(w, w.T)
l3 = DenseLayer(l2, num_units=500, W=w_psd)
{% endhighlight %}

## Limitations

There are only a couple of limitations to using Theano expressions as layer parameters. One is that Lasagne functions and methods such as `Layer.get_params()` will implicitly assume that any shared variable featuring in these Theano expressions is to be treated as a parameter. In practice that means you can't mix learnable and non-learnable parameter variables in a single expression. Also, the same tags will apply to all shared variables in an expression. More information about parameter tags can be found in [the documentation](http://lasagne.readthedocs.org/en/latest/modules/layers/base.html#lasagne.layers.Layer.get_params).

For almost all use cases, these limitations should not be an issue. If they are, your best bet is to implement a custom layer class. Luckily, [this is also very easy in Lasagne](http://lasagne.readthedocs.org/en/latest/user/custom_layers.html).

## Why it works

All of this is made possible because Lasagne builds on Theano, which takes care of backpropagating through the parameter expression to any underlying learned tensors. In frameworks building on hard-coded layer implementations rather than an automatic expression compiler, all these examples would require writing custom backpropagation code.

If you want to play around with this yourself, try the bleeding-edge version of Lasagne. You can find [installation instructions here](http://lasagne.readthedocs.org/en/latest/user/installation.html#bleeding-edge-version).

**Have fun experimenting!** If you've done something cool that you'd like to share, feel free to send us a pull request on our [Recipes repository](https://github.com/Lasagne/Recipes).

