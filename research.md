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

<!-- TODO: add an image from a paper  (spectrograms? k-means features?) -->

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

<!-- 
## What HPSTR brings to the table:

* Responsive templates for post, page, and post index `_layouts`. Looks great on mobile, tablet, and desktop devices.
* Gracefully degrads in older browsers. Compatible with Internet Explorer 8+ and all modern browsers.  
* Modern and minimal design.
* Sweet animated menu.
* Background image support.
* Readable typography to make your words shine.
* Support for large images to call out your favorite posts.
* Comments powered by [Disqus](http://disqus.com) if you choose to enable.
* Simple and clear permalink structure[^1].
* [Open Graph](https://developers.facebook.com/docs/opengraph/) and [Twitter Cards](https://dev.twitter.com/docs/cards) support for a better social sharing experience.
* Simple [custom 404 page]({{ site.url }}/404.html) to get you started.
* Stylesheets for Pygments and Coderay [syntax highlighting]({{ site.url }}/code-highlighting-post/) to make your code examples look snazzy
* [Grunt](http://gruntjs.com) build script for easy theme development

<div markdown="0"><a href="{{ site.url }}/theme-setup" class="btn btn-info">Install the Theme</a></div>

[^1]: Example: *domain.com/category-name/post-title* -->