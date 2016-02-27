---
layout: page
permalink: /research/
title: Research
tags: [research]
image:
  feature: 12.jpg
  <!--  credit: dargadgetz
  creditlink: http://www.dargadgetz.com/ios-7-abstract-wallpaper-pack-for-iphone-5-and-ipod-touch-retina/ -->
<!-- share: true -->
---

My main research interest is learning hierarchical representations of musical audio signals: finding ways to represent music audio to facilitate classification and recommendation by learning from data.

For this, I make use of feature learning and *[deep learning](http://en.wikipedia.org/wiki/Deep_learning)* techniques. I also use collaborative filtering techniques for music recommendation. Occasionally I venture outside of the realm of music and apply deep learning techniques to other types of data, such as images.

A few selected papers are listed below, please refer to Google Scholar for [an overview of my publications](http://scholar.google.be/citations?user=2ZU62T4AAAAJ).


### Exploiting cyclic symmetry in convolutional neural networks (submitted)

Sander Dieleman, Jeffrey De Fauw, Koray Kavukcuoglu

Many classes of images exhibit rotational symmetry. Convolutional neural networks are sometimes trained using data augmentation to exploit this, but they are still required to learn the rotation equivariance properties from the data. Encoding these properties into the network architecture could result in a more efficient use of the parameter budget by relieving the model from learning them. We introduce four operations which can be inserted into neural network models as layers, and which can be combined to make these models partially equivariant to rotations.

[**Paper (arXiv)**](http://arxiv.org/abs/1602.02660)

<figure>
    <a href="/images/cyclic_diagram.png"><img src="/images/cyclic_diagram.png" alt="Schematic representation of the effect of the proposed cyclic slice, roll and pool operations on the faeture maps in a convolutional neural network."></a>
    <figcaption>Schematic representation of the effect of the proposed cyclic slice, roll and pool operations on the faeture maps in a convolutional neural network.</figcaption>
</figure>



### Learning feature hierarchies for musical audio signals (PhD Thesis)

Sander Dieleman

This is my PhD thesis, which I defended in January 2016. It covers most of my work on applying deep learning to content-based music information retrieval. My work on galaxy morphology prediction is included as an appendix. Part of the front matter is in Dutch, but the main matter is in English.

[**Thesis (PDF)**](https://www.dropbox.com/s/22bqmco45179t7z/thesis-FINAL.pdf)



### Rotation-invariant convolutional neural networks for galaxy morphology prediction (MNRAS)

Sander Dieleman, Kyle W. Willett, Joni Dambre

I wrote a paper about my winning entry for the [Galaxy Challenge on Kaggle](http://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge), which I also [wrote about on this blog last year](http://benanne.github.io/2014/04/05/galaxy-zoo.html). In short, I trained convolutional neural networks for galaxy morphology prediction based on images, and made some modifications to the network architecture to exploit the rotational symmetry of the images. The paper was written together with one of the competition organizers and special attention is paid to how astronomers can actually benefit from this work.

[**Paper**](http://mnras.oxfordjournals.org/content/450/2/1441)

<figure>
    <a href="/images/architecture.png"><img src="/images/architecture.png" alt="Schematic diagram of the architecture of a convolutional network designed to exploit rotational symmetry in images of galaxies."></a>
    <figcaption>Schematic diagram of the architecture of a convolutional network designed to exploit rotational symmetry in images of galaxies.</figcaption>
</figure>



### End-to-end learning for music audio (ICASSP 2014)

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
