---
layout: post
title: "Diffusion language models"
description: "Diffusion models have completely taken over generative modelling of perceptual signals -- why is autoregression still the name of the game for language modelling? Can we do anything about that?"

tags: [diffusion, score function, deep learning, generative models, language]

image:
  feature: language.jpg
comments: true
share: true
---

Diffusion models have completely taken over generative modelling of perceptual signals such as images, audio and video. Why is autoregression still the name of the game for language modelling? And can we do anything about that? Some thoughts about what it will take for other forms of iterative refinement to take over language modelling, the last bastion of autoregression.

## <a name="diffusion"></a> The rise of diffusion models

<figure>
  <a href="/images/diffuse2.jpg"><img src="/images/diffuse2.jpg"></a>
</figure>

Roughly three years ago, things were starting to look as if adversarial image generators were about to be supplanted by a powerful combination of autoregression and discrete representation learning. [BigGAN](https://arxiv.org/abs/1809.11096)[^biggan] and [StyleGAN](https://arxiv.org/abs/1912.04958)[^stylegan] had significantly expanded the capabilities of image generators, but the mode-seeking nature of GANs made them favour realism over diversity. This presented some challenges, and people were having trouble reproducing impressive domain-specific results (e.g. generating realistic human faces) on more diverse training datasets.

[VQ-VAE 2](https://arxiv.org/abs/1906.00446)[^vqvae2] and especially [VQGAN](https://arxiv.org/abs/2012.09841)[^vqgan] extolled the virtue of a two-stage approach to generative modelling: first turn everything into a highly compressed discrete one-dimensional sequence, and then learn to predict this sequence step-by-step using a powerful autoregressive model. This idea had already proven fruitful before, going back to the original [VQ-VAE](https://arxiv.org/abs/1711.00937)[^vqvae], but these two papers really drove the point home that this was our best bet for generative modelling of diverse data at scale.

But then, a challenger appeared: a new generative modelling approach based on **iterative denoising** was starting to show promise. Yang Song and Stefano Ermon proposed score-based models: while their [NeurIPS 2019 paper](https://arxiv.org/abs/1907.05600)[^songermon] was more of a proof-of-concept, the next year's follow-up ['Improved Techniques for Training Score-Based Generative Models'](https://arxiv.org/abs/2006.09011)[^songermon2] showed results that convinced some people (including me!) to take this direction of research more seriously. Another NeurIPS 2020 paper by Jonathan Ho, Ajay Jain and Pieter Abbeel, ['Denoising Diffusion Probabilistic Models' (DDPMs)](https://arxiv.org/abs/2006.11239)[^ddpm] showed similar results, and it didn't take people too long to realise that DDPMs and score-based models were two sides of the same coin.

The real triumph of diffusion models over other alternatives for image generation came in 2021, with ['Diffusion Models Beat GANs on Image Synthesis'](https://arxiv.org/abs/2105.05233)[^beatgans] by Prafulla Dhariwal and Alex Nichol. At that point, it was pretty clear to everyone in the know that this approach was poised to take over. Powerful diffusion-based text-to-image models such as [GLIDE](https://arxiv.org/abs/2112.10741)[^glide] started to arrive by the end of that year, and proceeded to go mainstream in 2022.

**If you are unfamiliar with diffusion models, I recommend reading at least the first section of my previous blog post ['Diffusion models are autoencoders'](https://benanne.github.io/2022/01/31/diffusion.html#diffusion) for context, before reading the rest of this one.**

## <a name="match"></a> Diffusion for images: a match made in heaven

<figure>
  <a href="/images/noisy_mountains.jpg"><img src="/images/noisy_mountains.jpg" alt="A noisy image of a mountain range, with the level of noise gradually decreasing from left to right."></a>
</figure>

Diffusion models and the human visual system have one important thing in common: **they don't care too much about high frequencies**. At least, not out of the box. I discussed the reasons for this in some detail in [an earlier blog post](https://benanne.github.io/2022/01/31/diffusion.html#scale) (section 5 in particular).

In a nutshell, the different levels of noise at which a diffusion model operates allow it to focus on different spatial frequency components of the image at each iterative refinement step. When sampling an image, the model effectively builds it up from low frequencies to high frequencies, first filling in large-scale structure and then adding progressively more fine-grained details.

During training, we sample a noise level for each training example, add noise to it, and then try to predict the noise. The relative weights with which we sample the different noise levels therefore determine the degree to which the model focuses on large-scale and fine-grained structure. The most commonly used formulation, with uniform weighting of the noise levels, yields a very different objective than the likelihood loss which e.g. autoregressive models are trained with.

It turns out that there is a particular weighting which corresponds directly to the likelihood loss[^likelihood], but this puts significantly more weight on very low noise levels. Since low noise levels correspond to high spatial frequencies, this also indirectly explains why likelihood-based autoregressive models in pixel space never really took off: they end up spending way too much of their capacity on perceptually meaningless detail, and never get around to modelling larger-scale structure.

Relative to the likelihood loss, uniform weighting across noise levels in diffusion models yields an objective that is much more closely aligned with the human visual system. I don't believe this was actually known when people first started training diffusion models on images -- it was just a lucky coincidence! But we understand this pretty well now, and I think it is one of the two main reasons why this modelling approach completely took over in a matter of two years. (The other reason is of course **classifier-free guidance**, which you can read more about in [my previous blog post on the topic](https://benanne.github.io/2022/05/26/guidance.html).)

The reason I bring all this up here, is that **it doesn't bode particularly well for applications of diffusion models beyond the perceptual domain**. Our ears have a similar disdain for high frequencies as our eyes (though to a lesser extent, I believe), but in the language domain, what does "high frequency" even mean[^prism]? Given the success of likelihood-based language models, could the relatively lower weight of low noise levels actually prove to be a liability in this setting?

## <a name="ar"></a> Autoregression for language: a tough baseline to beat

<figure>
  <a href="/images/arguidance.jpg"><img src="/images/arguidance.jpg"></a>
</figure>

Autoregression at the word or token level is a very natural way to do language modelling, because to some degree, it reflects how language is produced and consumed: as a one-dimensional sequence, one element at a time, in a particular fixed order. However, if we consider the process through which an abstract thought turns into an utterance, the iterative denoising metaphor starts to look more appealing. When writing a paragraph, the core concepts are generally decided on first, and the exact wording and phrasing doesn't materialise until later. That said, perhaps it doesn't matter precisely how humans interact with language: just like how planes don't fly the same way birds do (h/t Yann LeCun), **the best way to build a practically useful language model need not reflect nature** either.

Practically speaking, autoregressive models have an interface that is somewhat limited: they can be *prompted*, i.e. tasked to complete a sequence for which a prefix is given. While this has actually been shown to be reasonably versatile in itself, the ability of non-autoregressive models to fill in the blanks (i.e. be conditioned on something other than a prefix, also known as inpainting in the image domain) is potentially quite useful, and not something that comes naturally to autoregressive models (though it is of course possible to do infilling with autoregressive models[^middle]).

### Training efficiency

If we compare autoregression and diffusion side-by-side as different forms of iterative refinement, the former has the distinct advantage that training can be parallelised trivially across all refinement steps. During autoregressive model training, we obtain a useful gradient signal from all steps in the sampling process. This is not true for diffusion models, where we have to sample a particular noise level for each training example. It is not practical to train on many different noise levels for each example, because that would require multiple forward and backward passes through the model. For autoregression, we get gradients for all sequence steps with just a single forward-backward pass.

As a result, **diffusion model training** is almost certainly significantly **less statistically efficient** than autoregressive model training, and slower convergence implies higher computational requirements.

### Sampling efficiency

Sampling algorithms for diffusion models are very flexible: they allow for sample quality and computational cost to be traded off without retraining, simply by changing the number of sampling steps. This isn't practical with autoregressive models, where the number of sampling steps is tied directly to the length of the sequence that is to be produced. On the face of it, diffusion models are at an advantage here: perhaps we can get high-quality samples with a number of steps that is significantly lower than the sequence length?

For long enough sequences, this is probably true, but it is important to compare apples to apples. Simply comparing the number of sampling steps across different methods relies on the implicit assumption that all sampling steps have the same cost, and this is not the case. Leaving aside the fact that a single diffusion sampling step can sometimes require multiple forward passes through the model, the cost of an individual forward pass also differs. Autoregressive models can benefit substantially from *caching*, i.e. re-use of activations computed during previous sampling steps, which significantly reduces the cost of each step. This is not the case for diffusion models, because the level of noise present in the input changes throughout sampling, so each sampling step requires a full forward pass across the entire input.

Therefore, the break-even point at which diffusion sampling becomes more efficient than autoregressive sampling is probably at a number of steps *significantly below* the length of the sequence. Whether this is actually attainable in practice remains to be seen.

### Why bother with diffusion at all?

The efficiency disadvantages with respect to autoregressive models might lead one to wonder if diffusion-based language modelling is even worth exploring to begin with. Aside from infilling capabilities and metaphorical arguments, there are a few other reasons why I believe it's worth looking into:

* Unlike autoregressive models, which require restricted connectivity patterns to ensure causality (usually achieved by masking), **diffusion model architectures are completely unconstrained**. This enables a lot more creative freedom, as well as potentially benefiting from architectural patterns that are common in other application domains, such as using pooling and upsampling layers to capture structure at multiple scales. One recent example of such creativity is Recurrent Interface Networks[^rins], whose Perceiver IO-like[^perceiverio] structure enables efficient re-use of computation across sampling steps.

* The **flexibility of the sampling procedure** extends beyond trading off quality against computational cost: it can also be modified to amplify the influence of conditioning signals (e.g. through classifier-free guidance), or to include additional constraints without retraining. Li et al.[^diffusionlm] extensively explore the latter ability for text generation (e.g. controlling sentiment or imposing a particular syntactic structure).

* Who knows what other perks we might uncover by properly exploring this space? The first few papers on diffusion models for images struggled to match results obtained with more established approaches at the time (i.e. GANs, autoregressive models). Work on diffusion models in new domains could follow the same trajectory -- **if we don't try, we'll never know**.


## <a name="discrete"></a> Diffusion for discrete data

<figure>
  <a href="/images/discrete.jpg"><img src="/images/discrete.jpg"></a>
</figure>

Diffusion models operate on continuous inputs by default. When using the score-based formalism, continuity is a requirement because the score function $$\nabla_\mathbf{x} \log p(\mathbf{x})$$ is only defined when $$\mathbf{x}$$ is continuous. Language is usually represented as a sequence of discrete tokens, so the standard formulation is not applicable. Broadly speaking, there are two ways to tackle this apparent incompatibility:

* formulate a **discrete corruption process** as an alternative to Gaussian diffusion;
* **map discrete inputs to continuous vectors** and apply Gaussian diffusion in that space.

The former approach has been explored extensively: D3PM[^d3pm], MaskGIT[^maskgit], Mask-predict[^maskpredict], ARDM[^ardm], Multinomial diffusion[^multinomial], DiffusER[^diffuser] and SUNDAE[^sundae] are all different flavours of non-autoregressive iterative refinement using a discrete corruption process. Many (but not all) of these works focus on language modelling as the target application. It should be noted that machine translation has been particularly fertile ground for this line of work, because the strong conditioning signal makes non-autoregressive methods attractive even when their ability to capture diversity is relatively limited. Several works on non-autoregressive machine translation predate the rise of diffusion models.

Unfortunately, moving away from the standard continuous formulation of diffusion models tends to mean giving up on some useful features, such as classifier-free guidance and the ability to use various accelerated sampling algorithms developed specifically for this setting. Luckily, we can stick with continuous Gaussian diffusion simply by **embedding** discrete data in Euclidean space. This approach has recently been explored for language modelling. Some methods, like self-conditioned embedding diffusion (SED)[^sed], use a separate representation learning model to obtain continuous embeddings corresponding to discrete tokens; others jointly fit the embeddings and the diffusion model, like Diffusion-LM[^diffusionlm], CDCD[^cdcd] and Difformer[^difformer].

[**Continuous diffusion for categorical data (CDCD)**](https://arxiv.org/abs/2211.15089) is my own work in this space: we set out to explore how diffusion models could be adapted for language modelling. One of the goals behind this research project was to develop a method for diffusion language modelling that looks as familiar as possible to language modelling practitioners. Training diffusion models is a rather different experience from training autoregressive Transformers, and we wanted to **minimise the differences to make this as approachable as possible**. The result is a model whose training procedure is remarkably close to that of BERT[^bert]: the input token sequence is embedded, noise is added to the embeddings, and the model learns to predict the original tokens using the cross-entropy loss (*score interpolation*). The model architecture is a standard Transformer. We address the issue of finding the right weighting for the different noise levels with an active learning strategy (*time warping*), which adapts the distribution of sampled noise levels on the fly during training.

Another way to do language modelling with Gaussian diffusion, which to my knowledge has not been explored extensively so far, is to **learn higher-level continuous representations** rather than embed individual tokens. This would require a powerful representation learning approach that learns representations that are rich enough to be decoded back into readable text (potentially by a light-weight autoregressive decoder). Autoencoders applied to token sequences tend to produce representations that fail to capture the least predictable components of the input, which carry precisely the most salient information. Perhaps contrastive methods, or methods that try to capture the dynamics of text (such as Time Control[^timecontrol]) could be more suitable for this purpose.

## <a name="closing-thoughts"></a> Closing thoughts

<figure>
  <a href="/images/sunset2.jpg"><img src="/images/sunset2.jpg"></a>
</figure>

While CDCD models produce reasonable samples, and are relatively easy to scale due to their similarity to existing language models, the efficiency advantages of autoregression make it a very tough baseline to beat. I believe it is still **too early to consider diffusion as a serious alternative to autoregression for generative language modelling at scale**.  As it stands, we also know next to nothing about scaling laws for diffusion models. Perhaps ideas such as latent self-conditioning[^rins] could make diffusion more competitive, by improving computational efficiency, but it's not clear that this will be sufficient. Further exploration of this space has the potential to pay off handsomely!

All in all, I have become convinced that the key to powerful generative models is **iterative refinement**: rather than generating a sample in a single pass through a neural network, the model is applied repeatedly to refine a canvas, and hence the unrolled sampling procedure corresponds to a much "deeper" computation graph. Exactly which algorithm one uses to achieve this might not matter too much in the end, whether it be autoregression, diffusion, or something else entirely. I have a lot more thoughts about this, so perhaps this could be the subject of a future blog post.

*On an unrelated note: I've disabled Disqus comments on all of my blog posts, as their ads seem to have gotten very spammy. I don't have a good alternative to hand right now, so in the meantime, feel free to tweet your thoughts at me instead [@sedielem](https://twitter.com/sedielem), or send me an email. When I eventually revamp this blog at some point in the future, I will look into re-enabling comments. Apologies for the inconvenience!*

*If you would like to cite this post in an academic context, you can use this BibTeX snippet:*

```
@misc{dieleman2023language,
  author = {Dieleman, Sander},
  title = {Diffusion language models},
  url = {https://benanne.github.io/2023/01/09/diffusion-language.html},
  year = {2023}
}
```

## <a name="acknowledgements"></a> Acknowledgements

Thanks to my collaborators on the CDCD project, and all my colleagues at DeepMind.

## <a name="references"></a> References

[^biggan]: Brock, Donahue, Simonyan, "[Large Scale GAN Training for High Fidelity Natural Image Synthesis](https://arxiv.org/abs/1809.11096)", International Conference on Learning Representations, 2019.

[^stylegan]: Karras, Laine, Aittala, Hellsten, Lehtinen, Aila, "[Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958)", Computer Vision and Pattern Recognition, 2020.

[^vqvae2]: Razavi, van den Oord and Vinyals, "[Generating Diverse High-Fidelity Images with VQ-VAE-2](https://arxiv.org/abs/1906.00446)", Neural Information Processing Systems, 2019.

[^vqgan]: Esser, Rombach and Ommer, "[Taming Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2012.09841)", Computer Vision and Pattern Recognition, 2021.

[^vqvae]: van den Oord, Vinyals and Kavukcuoglu, "[Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937)", Neural Information Processing Systems, 2017.

[^songermon]: Song and Ermon, "[Generative Modeling by Estimating Gradients of the Data Distribution](https://arxiv.org/abs/1907.05600)", Neural Information Processing Systems, 2019.

[^songermon2]: Song and Ermon, "[Improved Techniques for Training Score-Based Generative Models](https://arxiv.org/abs/2006.09011)", Neural Information Processing Systems, 2020.

[^ddpm]: Ho, Jain and Abbeel, "[Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)", Neural Information Processing Systems, 2020.

[^beatgans]: Dhariwal, Nichol, "[Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233)", Neural Information Processing Systems, 2021.

[^glide]: Nichol, Dhariwal, Ramesh, Shyam, Mishkin, McGrew, Sutskever, Chen, "[GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models](https://arxiv.org/abs/2112.10741)", arXiv, 2021.

[^likelihood]: Song, Durkan, Murray, Ermon, "[Maximum Likelihood Training of Score-Based Diffusion Models](https://arxiv.org/abs/2101.09258)", Neural Information Processing Systems, 2021.

[^prism]: Tamkin, Jurafsky, Goodman, "[Language Through a Prism: A Spectral Approach for Multiscale Language Representations](https://arxiv.org/abs/2011.04823)", Neural Information Processing Systems, 2020.

[^middle]: Bavarian, Jun, Tezak, Schulman, McLeavey, Tworek, Chen, "[Efficient Training of Language Models to Fill in the Middle](https://arxiv.org/abs/2207.14255)", arXiv, 2022.

[^rins]: Jabri, Fleet, Chen, "[Scalable Adaptive Computation for Iterative Generation](https://arxiv.org/abs/2212.11972)", arXiv, 2022.

[^perceiverio]: Jaegle, Borgeaud, Alayrac, Doersch, Ionescu, Ding, Koppula, Zoran, Brock, Shelhamer, Hénaff, Botvinick, Zisserman, Vinyals, Carreira, "[Perceiver IO: A General Architecture for Structured Inputs & Outputs](https://arxiv.org/abs/2107.14795)", International Conference on Learning Representations, 2022.

[^diffusionlm]: Li, Thickstun, Gulrajani, Liang, Hashimoto, "[Diffusion-LM Improves Controllable Text Generation](https://arxiv.org/abs/2205.14217)", Neural Information Processing Systems, 2022.

[^d3pm]: Austin, Johnson, Ho, Tarlow, van den Berg, "[Structured Denoising Diffusion Models in Discrete State-Spaces](https://arxiv.org/abs/2107.03006)", Neural Information Processing Systems, 2021.

[^maskgit]: Chang, Zhang, Jiang, Liu, Freeman, "[MaskGIT: Masked Generative Image Transformer](https://arxiv.org/abs/2202.04200)", Computer Vision and Patern Recognition, 2022.

[^maskpredict]: Ghazvininejad, Levy, Liu, Zettlemoyer, "[Mask-Predict: Parallel Decoding of Conditional Masked Language Models](https://arxiv.org/abs/1904.09324)", Empirical Methods in Natural Language Processing, 2019.

[^ardm]: Hoogeboom, Gritsenko, Bastings, Poole, van den Berg, Salimans, "[Autoregressive Diffusion Models](https://arxiv.org/abs/2110.02037)", International Conference on Learning Representations, 2022.

[^multinomial]: Hoogeboom, Nielsen, Jaini, Forré, Welling, "[Argmax Flows and Multinomial Diffusion: Learning Categorical Distributions](https://arxiv.org/abs/2102.05379)", Neural Information Processing Systems, 2021.

[^diffuser]: Reid, Hellendoorn, Neubig, "[DiffusER: Discrete Diffusion via Edit-based Reconstruction](https://arxiv.org/abs/2210.16886)", arXiv, 2022.

[^sundae]: Savinov, Chung, Binkowski, Elsen, van den Oord, "[Step-unrolled Denoising Autoencoders for Text Generation](https://arxiv.org/abs/2112.06749)", International Conference on Learning Representations, 2022.

[^sed]: Strudel, Tallec, Altché, Du, Ganin, Mensch, Grathwohl, Savinov, Dieleman, Sifre, Leblond, "[Self-conditioned Embedding Diffusion for Text Generation](https://arxiv.org/abs/2211.04236)", arXiv, 2022.

[^cdcd]: Dieleman, Sartran, Roshannai, Savinov, Ganin, Richemond, Doucet, Strudel, Dyer, Durkan, Hawthorne, Leblond, Grathwohl, Adler, "[Continuous diffusion for categorical data](https://arxiv.org/abs/2211.15089)", arXiv, 2022.

[^difformer]: Gao, Guo, Tan, Zhu, Zhang, Bian, Xu, "[Difformer: Empowering Diffusion Model on Embedding Space for Text Generation](https://arxiv.org/abs/2212.09412)", arXiv, 2022.

[^bert]: Devlin, Chang, Lee, Toutanova, "[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)", North American Chapter of the Association for Computational Linguistics, 2019.

[^timecontrol]: Wang, Durmus, Goodman, Hashimoto, "[Language modeling via stochastic processes](https://arxiv.org/abs/2203.11370)", International Conference on Learning Representations, 2022.
