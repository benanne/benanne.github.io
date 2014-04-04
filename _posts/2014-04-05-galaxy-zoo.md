---
layout: post
title: Solution for the Galaxy Zoo challenge
description: "My solution for the Galaxy Zoo challenge using convolutional neural networks"
<!-- modified: 2014-03-29 -->

tags: [deep learning, convolutional neural networks, convnets, Theano, Kaggle, Galaxy Zoo, competition]

<!-- image:
  feature: abstract-3.jpg
  credit: dargadgetz
  creditlink: http://www.dargadgetz.com/ios-7-abstract-wallpaper-pack-for-iphone-5-and-ipod-touch-retina/ -->

comments: true
share: true
---

The [Galaxy Zoo challenge](http://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge) on Kaggle has just finished. The goal of the competition was to predict how Galaxy Zoo users (*zooites*) would classify images of galaxies from the [Sloan Digital Sky Survey](http://www.sdss.org/).

In this post, I'm going to explain how my solution works. I will focus on what I used for my final submissions, and occasionally I'll mention some other things that I tried, but I will leave out some stuff for brevity's sake.

## The problem

[Galaxy Zoo](http://www.galaxyzoo.org/) is a crowdsourcing project, where users are asked to describe the morphology of galaxies based on images. They are asked questions such as "How rounded is the galaxy" and "Does it have a central bulge", and the users' answers determine which question will be asked next. The questions form a decision tree which is shown in the figure below, taken from [Willett et al. 2013](http://arxiv.org/abs/1308.3496).

<figure>
  <a href="/images/gzoo_decision_tree.png"><img src="/images/gzoo_decision_tree.png" alt=""></a>
  <figcaption>The Galaxy Zoo decision tree, taken from <a href="http://arxiv.org/abs/1308.3496">Willett et al. 2013</a>.</figcaption>
</figure>

When many users have classified the same image, their answers can be aggregated into a set of probabilities for each answer. Often, not all users will agree on all their answers, so it's useful to quantify this uncertainty.

The goal of the [Galaxy Zoo challenge](http://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge) is to predict these probabilities from the galaxy images that are shown to the users. In other words, build a model of how "the crowd" perceive and classify these images.

This means that we're looking at a *regression* problem, not a classification problem: we don't have to determine which classes the galaxies belong to, but rather the fraction of people who would classify them as such.




## Summary of the solution



* software used
* approach: convnets
* the key thing was to avoid overfitting
    * data augmentation
    * dropout
    * extra weight sharing in the architecture

## Preprocessing and data augmentation

* downsampling and augmentation in one go (it's all affine transformations)
* color augmentation
* other things tried: gaussian noise, ...

## Network architecture

* details about the network architecture (and some variations)
    * views/parts
    * maxout
    * make sure to include a drawing of the single best model.
    * output constraints

<figure>
  <a href="/images/schema_views_600.png"><img src="/images/schema_views_600.png" alt=""></a>
  <figcaption>Four different <strong>views</strong> are extracted from each image: a regular view (red), a 45° rotated view (green), and mirrored versions of both.</figcaption>
</figure>

<figure>
  <a href="/images/schema_parts_600.png"><img src="/images/schema_parts_600.png" alt=""></a>
  <figcaption>Each view is then split into four partially overlapping <strong>parts</strong>. Each part is rotated so that they are all aligned, with the galaxy in the bottom right corner. In total, 16 parts are extracted from the original image.</figcaption>
</figure>


<figure>
  <a href="/images/architecture.png"><img src="/images/architecture.png" alt=""></a>
  <figcaption>Krizhevsky-style diagram of the architecture of the best performing network.</figcaption>
</figure>

## Optimization

* stochastic gradient descent, nesterov momentum, learning rate schedule, details for the hyperparameters
* dropout and norm constraints

## Model averaging

* uniform blend across 60 transformations of the input
* uniform + weighted + separate weighted blend across a bunch of models

## Miscellany

* also polar representations near the end (nice because rotation invariance = translation invariance)
* input at multiple scales, this didn't work (no gradient for the other scale)
* ...
