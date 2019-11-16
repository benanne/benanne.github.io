---
layout: page
title: Software
tags: [sesearch]
image:
  feature: 12.jpg
comments: false
modified: 2019-11-16
---

## Lasagne

Lasagne is a lightweight library to build and train neural networks in Theano. Pragmatism and transparency are two of its main design goals. It's supposed to help you build networks *with* Theano, not on top of it, so it doesn't hide Theano behind abstractions.

Lasagne grew out of my internship at Spotify and my code for the Galaxy Challenge on Kaggle (see below as well). The core development team consists of [Daniel Nouri](http://danielnouri.org/), [Jan Schlüter](http://www.ofai.at/~jan.schlueter/), [Colin Raffel](http://colinraffel.com/), [Aäron van den Oord](https://twitter.com/avdnoord), [Eric Battenberg](http://ericbattenberg.com/), [Eben Olson](https://github.com/ebenolson), [Søren Kaae Sønderby](http://www1.bio.ku.dk/english/staff/?pure=en/persons/418078) and myself.

* [Code on GitHub](https://github.com/Lasagne/Lasagne)


## Kaggle plankton classification solution

Six of my colleagues and I won the first edition of the [National Data Science Bowl on Kaggle](https://www.kaggle.com/c/datasciencebowl), where the goal was to classify grayscale images of plankton according to species.

We used convolutional neural networks with architectures inspired by OxfordNet, and used some tricks to exploit the rotation invariance properties of the images. More information about our approach can be found [in this blogpost](http://benanne.github.io/2015/03/17/plankton.html). Documentation is also available in the GitHub repository.

* [Code on GitHub](https://github.com/benanne/kaggle-ndsb)


## Weighted matrix factorization

I spent some time implementing the weighted matrix factorization (WMF) algorithm by Hu, Koren and Volinsky (*Collaborative filtering for implicit feedback datasets*, ICDM 2008) both with numpy/scipy and with Theano. The Theano version ended up being hard to optimize, in most cases the numpy/scipy version is actually faster. Both implementations were not thoroughly tested and are not guaranteed to be correct. Use them at your own risk!

* [wmf on GitHub](https://github.com/benanne/wmf)
* [theano_wmf on GitHub](https://github.com/benanne/theano_wmf)

## Kaggle galaxy challenge solution

I won the [galaxy challenge on Kaggle](http://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge), which was organised by [Galaxy Zoo](http://www.galaxyzoo.org/). The goal was to predict how the crowd would classify images of galaxies, based on their morphology.

I used convolutional neural networks and incorporated the rotation invariance of the images into the network architecture. I also used lots of data augmentation. More information about my approach can be found [in this blogpost](http://benanne.github.io/2014/04/05/galaxy-zoo.html). Documentation is also available in the GitHub repository.

* [Code on GitHub](https://github.com/benanne/kaggle-galaxies)

## Morb

Morb is a toolbox for building and training Restricted Boltzmann Machine (RBM) models in [Theano](http://deeplearning.net/software/theano/). It is intended to be modular, so that a variety of different models can be built from their elementary parts. A second goal is for it to be extensible, so that new algorithms and techniques can be plugged in easily.

RBM implementations typically focus on one particular aspect of the model; for example, one implementation supports softmax units, another supports a different learning algorithm like fast persistent contrastive divergence (FPCD), yet another has convolutional weights, ... but finding an implementation of a convolutional softmax-RBM trained with FPCD is a lot more challenging :)

With Morb, I tried to tackle this issue by separating the implementation of different unit types, different parameterisations and different learning algorithms. That way, they all can be combined with each other, and using multiple unit types and parameter types together in a single model is also possible.

I am no longer working on this since I'm not really using RBMs in my research anymore.

<figure>
    <a href="https://github.com/benanne/morb"><img src="/images/morblogo.png"></a>
</figure>

* [Code on GitHub](https://github.com/benanne/morb)

## Kaggle whale detection challenge solution

In 2013 there was a [whale detection challenge on Kaggle](http://www.kaggle.com/c/whale-detection-challenge) which I participated in. The goal was to detect right whale calls in underwater recordings.

I got started very late, so I only had a few days to spend on this, which influenced my choice of approach: I went with a solution based on unsupervised feature learning with the spherical K-means algorithm, because it's very fast.

I was able to train a single model and do predictions with it in about half an hour, which allowed me to train a large number of these models in a random parameter search, which I had running continuously on a bunch of computers. My final solution, which got me to the 8th spot in the ranking, was an average of about 30 of these models.

The processing pipeline is roughly as follows: 

* **2x downsampling**: high frequencies are not relevant for this problem, so this helped to reduce the dimensionality of the input.
* **Spectrogram extraction**: I extracted spectrograms with a linear frequency scale using matplotlib's 'specgram' function, and then applied logarithmic scaling of the form f(x) = log(1 + C*x), where C is a parameter that was included in the random search.
* **Local normalisation**: for this step, the spectrograms were treated as images and local contrast normalisation was applied, to account for differences in volume (sometimes the noise in the samples was much louder than the whale call).
* **Patch extraction and whitening**: a bunch of patches were extracted from the spectrograms to learn features from them. They were first PCA-whitened.
* **Feature learning**: features were learnt on the whitened patches using the spherical K-means algorithm.
* **Feature extraction**: the features learnt from patches were convolved with the spectrograms, and then the output of this convolution was max-pooled across time and frequency, so that an output would be active if the feature was detected anywhere in the spectrogram.
* **Classifier training**: finally, a classifier was trained on these features. I tried SVMs, random forests and gradient boosting machines. I got the best results with random forests in the end (also taking in account execution time, because it had to be fast).

Hyperparameters for all these steps were optimised in a big random search. At the end I also added a bias to the predictions because it was discovered that the recordings were chronologically ordered. Exploiting this information gave another small score boost.

The processing pipeline is in the file [pipeline_job.py](https://github.com/benanne/kaggle-whales/blob/master/pipeline_job.py). My implementation of spherical K-means is in [kmeans.py](https://github.com/benanne/kaggle-whales/blob/master/kmeans.py).

* [Code on GitHub](https://github.com/benanne/kaggle-whales)
