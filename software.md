---
layout: page
permalink: /software/
title: Software
tags: [software]
image:
  feature: 12.jpg
  <!--  credit: dargadgetz
  creditlink: http://www.dargadgetz.com/ios-7-abstract-wallpaper-pack-for-iphone-5-and-ipod-touch-retina/ -->
<!-- share: true -->
---

## Morb

Morb is a toolbox for building and training Restricted Boltzmann Machine (RBM) models in [Theano](http://deeplearning.net/software/theano/). It is intended to be modular, so that a variety of different models can be built from their elementary parts. A second goal is for it to be extensible, so that new algorithms and techniques can be plugged in easily.

RBM implementations typically focus on one particular aspect of the model; for example, one implementation supports softmax units, another supports a different learning algorithm like fast persistent contrastive divergence (FPCD), yet another has convolutional weights, ... but finding an implementation of a convolutional softmax-RBM trained with FPCD is a lot more challenging :)

With Morb, I tried to tackle this issue by separating the implementation of different unit types, different parameterisations and different learning algorithms. That way, they all can be combined with each other, and using multiple unit types and parameter types together in a single model is also possible.

Documentation is limited for now, but this is a work in progress.

<figure>
    <a href="https://github.com/benanne/morb"><img src="/images/morblogo.png"></a>
</figure>

[Code on GitHub](https://github.com/benanne/morb)

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

[Code on GitHub](https://github.com/benanne/kaggle-whales)

<!-- 

My main research interest is learning hierarchical representations of musical audio signals: finding ways to represent music audio to facilitate classification and recommendation by learning from data.


For this, I make use of feature learning and *[deep learning](http://en.wikipedia.org/wiki/Deep_learning)* techniques. I also use collaborative filtering techniques for music recommendation. A few selected papers are listed below, please refer to Google Scholar for [an overview of my publications](http://scholar.google.be/citations?user=2ZU62T4AAAAJ).

### End-to-end learning for music audio (ICASSP 2014, to appear)

Sander Dieleman, Benjamin Schrauwen

Content-based music information retrieval tasks have traditionally been solved using engineered features and shallow processing architectures. In recent years, there has been increasing interest in using feature learning and deep architectures instead, thus reducing the required engineering
effort and the need for prior knowledge. However, this new approach typically still relies on mid-level representations of music audio, e.g. spectrograms, instead of raw audio signals. In this paper, we investigate whether it is possible to train convolutional neural networks directly on raw audio signals. The networks are able to autonomously discover frequency decompositions from raw audio, as well as phase- and translation-invariant feature representations.

[**Paper (PDF)**](https://dl.dropboxusercontent.com/u/19706734/paper_pt.pdf)

[copyright 2014 by IEEE](/ieee_copyright/)

<figure class='half'>
    <a href="/images/sorted_features_cropped.png"><img src="/images/sorted_features_cropped.png" alt="Normalised magnitude spectra of the filters learned in the lowest layer of a convolutional neural network that processes raw audio signals, ordered according to the dominant frequency (from low to high)."></a>
    <a href="/images/some_invariance_filters_cropped.png"><img src="/images/some_invariance_filters_cropped.png" alt="A subset of filters learned in a convolutional neural network with a feature pooling layer (L2 pooling with pools of 4 filters)."></a>
    <figcaption><strong>Left:</strong> normalised magnitude spectra of the filters learned in the lowest layer of a convolutional neural network that processes raw audio signals, ordered according to the dominant frequency (from low to high). <strong>Right:</strong> a subset of filters learned in a convolutional neural network with a feature pooling layer (L2 pooling with pools of 4 filters). Each row represents a filter group. The filters were low-pass filtered to remove noise and make the dominant frequencies stand out.</figcaption>
</figure>



### Deep content-based music recommendation (NIPS 2013)

Aäron van den Oord, Sander Dieleman, Benjamin Schrauwen

The collaborative filtering approach to music recommendation suffers from the cold start problem: it fails when no listening data is available, so it is not effective for recommending new and unpopular songs. In this paper, we use a latent factor model for recommendation, and predict the latent factors from music audio when they cannot be obtained from listening data, using a deep convolutional neural network. Predicted latent factors produce sensible recommendations, despite the fact that there is a large semantic gap between the characteristics of a song that affect user preference and the corresponding audio signal.

[**Paper (PDF)**](http://papers.nips.cc/paper/5004-deep-content-based-music-recommendation.pdf) - [**BibTeX**](http://papers.nips.cc/paper/5004-deep-content-based-music-recommendation/bibtex) - [**Abstract**](http://papers.nips.cc/paper/5004-deep-content-based-music-recommendation)

<figure>
    <a href="/images/prentje_nips.png"><img src="/images/prentje_nips.png" alt="t-SNE visualisation of user listening patterns predicted from audio."></a>
    <figcaption>t-SNE visualisation of user listening patterns predicted from audio. A few close-ups show artists whose songs are projected in specific areas.</figcaption>
</figure>



### Multiscale approaches to music audio feature learning (ISMIR 2013)

Sander Dieleman, Benjamin Schrauwen

Recent results in feature learning indicate that simple algorithms such as K-means can be very effective, sometimes surpassing more complicated approaches based on restricted Boltzmann machines, autoencoders or sparse coding. Furthermore, there has been increased interest in multiscale representations of music audio recently. Such representations are more versatile because music audio exhibits structure on multiple timescales, which are relevant for different MIR tasks to varying degrees. We develop and compare three approaches to multiscale audio feature learning using the spherical K-means algorithm.

[**Paper (PDF)**](http://www.ppgia.pucpr.br/ismir2013/wp-content/uploads/2013/09/69_Paper.pdf) - [**BibTeX**](http://dc.ofai.at/browser?b=1250)

<figure class='third'>
    <a href="/images/multires_cropped.png"><img src="/images/multires_cropped.png" alt="Multiresolution spectrograms"></a>
    <a href="/images/pyramid_gaussian_cropped.png"><img src="/images/pyramid_gaussian_cropped.png" alt="Gaussian pyramid"></a>
    <a href="/images/pyramid_laplacian_cropped.png"><img src="/images/pyramid_laplacian_cropped.png" alt="Laplacian pyramid"></a>

    <figcaption>Three multiscale time-frequency representations of audio signals. From left to right: multiresolution spectrograms, Gaussian pyramid, Laplacian pyramid.</figcaption>
</figure>

 -->