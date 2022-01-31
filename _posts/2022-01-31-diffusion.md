---
layout: post
title: Diffusion models are autoencoders
description: "Diffusion models have become very popular over the last two years. There is an underappreciated link between diffusion models and autoencoders."

tags: [diffusion, denoising, autoencoders, score function, deep learning, generative models]

image:
  feature: diffuse.jpg
comments: true
share: true
---


Diffusion models took off like a rocket at the end of 2019, after the publication of Song & Ermon's [seminal paper](https://arxiv.org/abs/1907.05600). In this blog post, I highlight a connection to another type of model: the venerable autoencoder.

## <a name="diffusion"></a> Diffusion models

<figure>
  <a href="/images/diffuse2.jpg"><img src="/images/diffuse2.jpg"></a>
</figure>

Diffusion models are fast becoming the go-to model for any task that requires producing perceptual signals, such as images and sound. They provide similar fidelity as alternatives based on generative adversarial nets (GANs) or autoregressive models, but with much better mode coverage than the former, and a faster and more flexible sampling procedure compared to the latter.

In a nutshell, diffusion models are constructed by first describing a procedure for gradually turning data into noise, and then training a neural network that learns to invert this procedure step-by-step. Each of these steps consists of **taking a noisy input and making it slightly less noisy**, by filling in some of the information obscured by the noise. If you start from pure noise and do this enough times, it turns out you can generate data this way!

Diffusion models have been around for a while[^equilibrium], but really took off at the end of 2019[^songermon]. The ideas are young enough that the field hasn't really settled on one particular convention or paradigm to describe them, which means almost every paper uses a slightly different framing, and often a different notation as well. This can make it quite challenging to see the bigger picture when trawling through the literature, of which there is already a lot! Diffusion models go by many names: *denoising diffusion probabilistic models* (DDPMs)[^ddpm], *score-based generative models*, or *generative diffusion processes*, among others. Some people just call them *energy-based models* (EBMs), of which they technically are a special case.

My personal favourite perspective starts from the idea of *score matching*[^scorematching] and uses a formalism based on stochastic differential equations (SDEs)[^sde]. For an in-depth treatment of diffusion models from this perspective, I strongly recommend [Yang Song's richly illustrated blog post](https://yang-song.github.io/blog/2021/score/) (which also comes with code and colabs). It is especially enlightening with regards to the connection between all these different perspectives. If you are familiar with variational autoencoders, you may find [Lilian Weng](https://lilianweng.github.io/lil-log/2021/07/11/diffusion-models.html) or [Jakub Tomczak](https://jmtomczak.github.io/blog/10/10_ddgms_lvm_p2.html)'s takes on this model family more approachable. 

If you are curious about generative modelling in general, [section 3 of my blog post](https://benanne.github.io/2020/03/24/audio-generation.html#generative-models) on generating music in the waveform domain contains a brief overview of some of the most important concepts and model flavours.


## <a name="autoencoders"></a> Denoising autoencoders

<figure>
  <a href="/images/bottleneck.jpg"><img src="/images/bottleneck.jpg"></a>
</figure>

Autoencoders are neural networks that are trained to predict their input. In and of itself, this is a trivial and meaningless task, but it becomes much more interesting when the network architecture is restricted in some way, or when the input is corrupted and the network has to learn to undo this corruption.

A typical architectural restriction is to introduce some sort of **bottleneck**, which limits the amount of information that can pass through. This implies that the network must learn to encode the most important information efficiently to be able to pass it through the bottleneck, in order to be able to accurately reconstruct the input. Such a bottleneck can be created by reducing the capacity of a particular layer of the network, by introducing quantisation (as in VQ-VAEs[^vqvae]) or by applying some form of regularisation to it during training (as in VAEs[^vaekingma] [^vaerezende] or contractive autoencoders[^cae]). The internal representation used in this bottleneck (often referred to as the *latent representation*) is what we are really after. **It should capture the essence of the input, while discarding a lot of irrelevant detail.**

Corrupting the input is another viable strategy to make autoencoders learn useful representations. One could argue that models with corrupted input are not autoencoders in the strictest sense, because the input and target output differ, but this is really a semantic discussion -- one could just as well consider the corruption procedure part of the model itself. In practice, such models are typically referred to as *denoising autoencoders*.

Denoising autoencoders were actually some of the first true "deep learning" models: back when we hadn't yet figured out how to reliably train neural networks deeper than a few layers with simple gradient descent, the prevalent approach was to pre-train networks layer by layer, and denoising autoencoders were frequently used for this purpose[^sdae] (especially by Yoshua Bengio and colleagues at MILA -- restricted Boltzmann machines were another option, favoured by Geoffrey Hinton and colleagues).

## <a name="peas"></a> One and the same?

<figure>
  <a href="/images/spiderman.jpg"><img src="/images/spiderman.jpg"></a>
</figure>

**So what is the link between modern diffusion models and these -- by deep learning standards -- ancient autoencoders?** I was inspired to ponder this connection a bit more after seeing some recent tweets speculating about autoencoders making a comeback:

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Are autoencoders making / going to make a comeback?</p>&mdash; David Krueger (@DavidSKrueger) <a href="https://twitter.com/DavidSKrueger/status/1428403382293876743?ref_src=twsrc%5Etfw">August 19, 2021</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Can you bring autoencoders back by the time my book is out, I&#39;m aiming for 2023</p>&mdash; Peli Grietzer (@peligrietzer) <a href="https://twitter.com/peligrietzer/status/1487186529999069186?ref_src=twsrc%5Etfw">January 28, 2022</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

As far as I'm concerned, **the autoencoder comeback is already in full swing, it's just that we call them diffusion models now!** Let's unpack this.


The neural network that makes diffusion models tick is trained to estimate the so-called *score function*, $$\nabla_\mathbf{x} \log p(\mathbf{x})$$, the gradient of the log-likelihood w.r.t. the input (a vector-valued function): $$\mathbf{s}_\theta (\mathbf{x}) = \nabla_\mathbf{x} \log p_\theta(\mathbf{x})$$. Note that this is different from $$\nabla_\theta \log p_\theta(\mathbf{x})$$, the gradient w.r.t. the model parameters $$\theta$$, which is the one you would use for training if this were a likelihood-based model. The latter tells you how to change the model parameters to increase the likelihood of the input under the model, whereas the former tells you how to *change the input itself* to increase its likelihood. (This is the same gradient you would use for DeepDream-style manipulation of images.)

In practice, we want to use the same network at every point in the gradual denoising process, i.e. at every noise level (from pure noise all the way to clean data). To account for this, it takes an additional input $$t \in [0, 1]$$ which indicates how far along we are in the denoising process: $$\mathbf{s}_\theta (\mathbf{x}_t, t) = \nabla_{\mathbf{x}_t} \log p_\theta(\mathbf{x}_t)$$. By convention, $$t = 0$$ corresponds to clean data and $$t = 1$$ corresponds to pure noise, so we actually "go back in time" when denoising.

The way you train this network is by taking inputs $$\mathbf{x}$$ and corrupting them with additive noise $$\mathbf{\varepsilon}_t \sim \mathcal{N}(0, \sigma_t^2)$$, and then predicting $$\mathbf{\varepsilon}_t$$ from $$\mathbf{x}_t = \mathbf{x} + \mathbf{\varepsilon}_t$$. The reason why this works is not entirely obvious. I recommend reading Pascal Vincent's 2010 tech report on the subject[^vincent] for an in-depth explanation of why you can do this.

Note that the variance depends on $$t$$, because it corresponds to the specific noise level at time $$t$$. The loss function is typically just mean squared error, sometimes weighted by a scale factor $$\lambda(t)$$, so that some noise levels are prioritised over others:

$$\arg\min_\theta \mathcal{L}_\theta = \arg\min_\theta \mathbb{E}_{t,p(\mathbf{x}_t)} \left[\lambda(t) ||\mathbf{s}_\theta (\mathbf{x} + \mathbf{\varepsilon}_t, t) - \mathbf{\varepsilon}_t||_2^2\right] .$$

Going forward, let's assume $$\lambda(t) \equiv 1$$, which is usually what is done in practice anyway (though other choices have their uses as well[^maxlikelihood]).

One key observation is that **predicting $$\mathbf{\varepsilon}_t$$ or $$\mathbf{x}$$ are equivalent**, so instead, we could just use

$$\arg\min_\theta \mathbb{E}_{t,p(\mathbf{x}_t)} \left[||\mathbf{s}_\theta' (\mathbf{x} + \mathbf{\varepsilon}_t, t) - \mathbf{x}||_2^2\right] .$$

To see that they are equivalent, consider taking a trained model $$\mathbf{s}_\theta$$ that predicts $$\mathbf{\varepsilon}_t$$ and add **a new residual connection** to it, going all the way from the input to the output, with a scale factor of $$-1$$. This modified model then predicts:

$$\mathbf{\varepsilon}_t - \mathbf{x}_t = \mathbf{\varepsilon}_t - (\mathbf{x} + \mathbf{\varepsilon}_t) = - \mathbf{x} . $$

In other words, we obtain a denoising autoencoder (up to a minus sign). This might seem surprising, but intuitively, it actually makes sense that **to increase the likelihood of a noisy input, you should probably just try to remove the noise, because noise is inherently unpredictable**. Indeed, it turns out that these two things are equivalent.


## <a name="tenuous"></a> A tenuous connection?

<figure>
  <a href="/images/bridge.jpg"><img src="/images/bridge.jpg"></a>
</figure>

Of course, the title of this blog post is intentionally a bit facetious: while there is a deeper connection between diffusion models and autoencoders than many people realise, the models have completely different purposes and so are not interchangeable.

**There are two key differences** with the denoising autoencoders of yore:
- the additional input $$t$$ makes one single model able to handle **many different noise levels** with a single set of shared parameters;
- we care about the output of the model, not the internal latent representation, so there is **no need for a bottleneck**. In fact, it would probably do more harm than good.

In the strictest sense, both of these differences have no bearing on whether the model can be considered an autoencoder or not. In practice, however, the point of an autoencoder is usually understood to be to learn a useful latent representation, so saying that diffusion models are autoencoders could perhaps be considered a bit... pedantic. Nevertheless, I wanted to highlight this connection because I think many more people know the ins and outs of autoencoders than diffusion models at this point. I believe that appreciating the link between the two can make the latter less daunting to understand.

This link is not merely a curiosity, by the way; it has also been the subject of several papers, which constitute an **early exploration of the ideas that power modern diffusion models**. Apart from the work by Pascal Vincent mentioned earlier[^vincent], there is also a series of papers by Guillaume Alain and colleagues[^gyom1] that[^gyom2] are[^gyom3] worth[^gyom4] checking[^gyom5] out[^gyom6]!

*[Note that there is another way to connect diffusion models to autoencoders, by viewing them as (potentially infinitely) deep latent variable models. I am personally less interested in that connection because it doesn't provide me with much additional insight, but it is just as valid. [Here's a blog post by Angus Turner](https://angusturner.github.io/generative_models/2021/06/29/diffusion-probabilistic-models-I.html) that explores this interpretation in detail.]*

## <a name="scale"></a> Noise and scale

<figure>
  <a href="/images/noisy_mountains.jpg"><img src="/images/noisy_mountains.jpg" alt="A noisy image of a mountain range, with the level of noise gradually decreasing from left to right."></a>
</figure>

I believe the idea of training a **single model to handle many different noise levels with shared parameters** is ultimately the key ingredient that made diffusion models really take off. Song & Ermon[^songermon] called them *noise-conditional score networks* (NCSNs) and provide a very lucid explanation of why this is important, which I won't repeat here.

The idea of using different noise levels in a single denoising autoencoder had previously been explored for representation learning, but not for generative modelling. Several works suggest gradually decreasing the level of noise over the course of training to improve the learnt representations[^geras1] [^chandra] [^zhang]. Composite denoising autoencoders[^geras2] have multiple subnetworks that handle different noise levels, which is a step closer to the score networks that we use in diffusion models, though still missing the parameter sharing.

A particularly interesting observation stemming from these works, which is also highly relevant to diffusion models, is that **representations learnt using different noise levels tend to correspond to different scales of features**: the higher the noise level, the larger-scale the features that are captured. I think this connection is worth investigating further: it implies that diffusion models fill in missing parts of the input at progressively smaller scales, as the noise level decreases step by step. This does seem to be the case in practice, and it is potentially a useful feature. Concretely, it means that $$\lambda(t)$$ can be designed to prioritise the modelling of particular feature scales! This is great, because excessive attention to detail is actually a major problem with likelihood-based models (I've previously discussed this in more detail in [section 6 of my blog post about typicality](https://benanne.github.io/2020/09/01/typicality.html#right-level)).

This connection between noise levels and feature scales was initially baffling to me: the noise $$\mathbf{\varepsilon}_t$$ that we add to the input during training is isotropic Gaussian, so **we are effectively adding noise to each input element (e.g. pixel) independently**. If that is the case, **how can the level of noise (i.e. the variance) possibly impact the scale of the features that are learnt?** I found it helpful to think of it this way:
- Let's say we are working with images. Each pixel in an image that could be part of a particular feature (e.g. a human face) provides **evidence for the presence (or absence) of that feature**.
- When looking at an image, **we implicitly aggregate the evidence** provided by all the pixels to determine which features are present (e.g. whether there is a face in the image or not).
- Larger-scale features in the image will cover a larger proportion of pixels. Therefore, **if a larger-scale feature is present** in an image, there is **more evidence** pointing towards that feature.
- Even if we add noise with a very high variance, that evidence will still be apparent, because **when combining information from all pixels, we average out the noise**.
- If more pixels are involved in this process, the tolerable noise level increases, because the maximal variance that still allows for the noise to be canceled out is much higher. For smaller-scale features however, recovery will be impossible because the noise dominates when we can only aggregate information from a smaller set of pixels.

Concretely, if an image contains a human face and we add a lot of noise to it, we will probably no longer be able to discern the face if it is far away from the camera (i.e. covers fewer pixels in the image), whereas if it is close to the camera, we might still see a faint outline. The header image of this section provides another example: the level of noise decreases from left to right. On the very left, we can still see the rough outline of a mountain despite very high levels of noise.



This is completely handwavy, but it provides some intuition for why there is a direct correspondence between the variance of the noise and the scale of features captured by denoising autoencoders and score networks.


## <a name="thoughts"></a> Closing thoughts

<figure>
  <a href="/images/sunset.jpg"><img src="/images/sunset.jpg"></a>
</figure>

So there you have it: **diffusion models are autoencoders. Sort of. When you squint a bit.** Here are some key takeaways, to wrap up:

- Learning to predict the score function $$\nabla_\mathbf{x} \log p(\mathbf{x})$$ of a distribution can be achieved by learning to denoise examples of that distribution. This is a core underlying idea that powers modern diffusion models.
- Compared to denoising autoencoders, score networks in diffusion models can handle all noise levels with a single set of parameters, and do not have bottlenecks. But other than that, they do the same thing.
- Noise levels and feature scales are closely linked: high noise levels lead to models capturing large-scale features, low noise levels lead to models focusing on fine-grained features.


*If you would like to cite this post in an academic context, you can use this BibTeX snippet:*

```
@misc{dieleman2022diffusion,
  author = {Dieleman, Sander},
  title = {Diffusion models are autoencoders},
  url = {https://benanne.github.io/2022/01/31/diffusion.html},
  year = {2022}
}
```

## <a name="acknowledgements"></a> Acknowledgements

Thanks to Conor Durkan and Katie Millican for fruitful discussions!

## <a name="references"></a> References

[^equilibrium]: Sohl-Dickstein, Weiss, Maheswaranathan and Ganguli, "[Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585)", International Conference on Machine Learning, 2015.

[^songermon]: Song and Ermon, "[Generative Modeling by Estimating Gradients of the Data Distribution](https://arxiv.org/abs/1907.05600)", Neural Information Processing Systems, 2019.

[^ddpm]: Ho, Jain and Abbeel, "[Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)", Neural Information Processing Systems, 2020.

[^sde]: Song, Sohl-Dickstein, Kingma, Kumar, Ermon and Poole, "[Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456)", International Conference on Learning Representations, 2021.

[^scorematching]: Hyvarinen, "[Estimation of Non-Normalized Statistical Models by Score Matching](https://www.jmlr.org/papers/v6/hyvarinen05a.html)", Journal of Machine Learning Research, 2005.

[^vqvae]: van den Oord, Vinyals and Kavukcuoglu, "[https://arxiv.org/abs/1711.00937](https://arxiv.org/abs/1711.00937)", Neural Information Processing Systems, 2017.

[^vaekingma]: Kingma and Welling, "[Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)", International Conference on Learning Representations, 2014.

[^vaerezende]: Rezende, Mohamed and Wierstra, "[Stochastic Backpropagation and Approximate Inference in Deep Generative Models](https://arxiv.org/abs/1401.4082)", International Conference on Machine Learning, 2014.

[^cae]: Rifai, Vincent, Muller, Glorot and Bengio, "[Contractive Auto-Encoders: Explicit Invariance During Feature Extraction](https://openreview.net/forum?id=HkZN5j-dZH)", International Conference on Machine Learning, 2011.

[^sdae]: Vincent, Larochelle, Lajoie, Bengio and Manzagol, "[Stacked Denoising Autoencoders: Learning Useful Representations in a Deep Network with a Local Denoising Criterion](https://www.jmlr.org/papers/v11/vincent10a.html)", Journal of Machine Learning Research, 2010.

[^maxlikelihood]: Song, Durkan, Murray and Ermon, "[Maximum Likelihood Training of Score-Based Diffusion Models](https://arxiv.org/abs/2101.09258)", Neural Information Processing Systems, 2021.

[^vincent]: Vincent, "[A Connection Between Score Matching and Denoising Autoencoders](http://www.iro.umontreal.ca/~vincentp/Publications/smdae_techreport.pdf)", Technical report, 2010.

[^gyom1]: Bengio, Alain and Rifai, "[Implicit density estimation by local moment matching to sample from auto-encoders](https://arxiv.org/abs/1207.0057)", arXiv, 2012.

[^gyom2]: Alain, Bengio and Rifai, "[Regularized auto-encoders estimate local statistics](http://www.eng.uwaterloo.ca/~jbergstr/files/nips_dl_2012/Paper%2029.pdf)", Neural Information Processing Systems, Deep Learning workshop, 2012.

[^gyom3]: Bengio,Yao, Alain and Vincent, "[Generalized denoising auto-encoders as generative models](https://arxiv.org/abs/1305.6663)", Neural Information Processing Systems, 2013.

[^gyom4]: Alain and Bengio, "[What regularized auto-encoders learn from the data-generating distribution](https://jmlr.org/papers/volume15/alain14a/alain14a.pdf)", Journal of Machine Learning Research, 2014.

[^gyom5]: Bengio, Laufer, Alain and Yosinski, "[Deep generative stochastic networks trainable by backprop](http://proceedings.mlr.press/v32/bengio14.pdf)", International Conference on Machine Learning, 2014.

[^gyom6]: Alain, Bengio, Yao, Yosinski, Laufer, Zhang and Vincent, "[GSNs: generative stochastic networks](https://arxiv.org/abs/1503.05571)", Information and Inference, 2016.

[^geras1]: Geras and Sutton, "[Scheduled denoising autoencoders](https://arxiv.org/abs/1406.3269)", International Conference on Learning Representations, 2015.

[^geras2]: Geras and Sutton, "[Composite denoising autoencoders](https://link.springer.com/chapter/10.1007/978-3-319-46128-1_43)", Joint European Conference on Machine Learning and Knowledge Discovery in Databases, 2016.

[^chandra]: Chandra and Sharma, "[Adaptive noise schedule for denoising autoencoder](https://link.springer.com/chapter/10.1007/978-3-319-12637-1_67)", Neural Information Processing Systems, 2014.

[^zhang]: Zhang and Zhang, "[Convolutional adaptive denoising autoencoders for hierarchical feature extraction](https://dl.acm.org/doi/abs/10.1007/s11704-016-6107-0)", Frontiers of Computer Science, 2018.

