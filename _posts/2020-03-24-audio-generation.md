---
layout: post
title: Generating music in the waveform domain
description: "This is a write-up of a presentation on generating music in the waveform domain, which was part of a tutorial that I co-presented at ISMIR 2019 earlier this month."

tags: [music, audio, waveform, raw audio, deep learning, generative models]

image:
  feature: musicbg2.jpg
comments: true
share: true
---

In November last year, I co-presented a tutorial on **waveform-based music processing with deep learning** with [Jordi Pons](http://www.jordipons.me/) and [Jongpil Lee](https://jongpillee.github.io/) at [ISMIR 2019](https://ismir2019.ewi.tudelft.nl/). Jongpil and Jordi talked about music classification and source separation respectively, and I presented the last part of the tutorial, on music generation in the waveform domain. It was very well received, so I've decided to write it up in the form of a blog post.

<div style="float: right; width: 30%;"><a href="https://ismir2019.ewi.tudelft.nl/"><img src="/images/ismir_logo.jpg" alt="ISMIR"></a></div>

ISMIR used to be my home conference when I was a PhD student working on music information retrieval, so it was great to be back for the first time in five years. With about 450 attendees (the largest edition yet), it made for a very different experience than what I'm used to with machine learning conferences like ICML, NeurIPS and ICLR, whose audiences tend to number in the thousands these days.

Our tutorial on the first day of the conference gave rise to plenty of interesting questions and discussions throughout, which inspired me to write some of these things down and hopefully provide a basis to continue these discussions online. Note that I will only be covering music generation in this post, but Jordi and Jongpil are working on blog posts about their respective parts. I will share them here when they are published. In the meantime, **the slide deck we used includes all three parts and is now available on [Zenodo (PDF)](https://zenodo.org/record/3529714#.XdBi0dv7Sf5) and on [Google slides](https://docs.google.com/presentation/d/1_ezZXDkyhp9USAYMc5oKJCkUrUhBfo-Di8H8IfypGBM/edit#slide=id.g647f5a8648_0_57)**.  I've also added a few things to this post that I've thought of since giving the tutorial, and some new work that has come out since.

This is also an excellent opportunity to revive my blog, which has lain dormant for the past four years. I have taken the time to update the blog software, so if anything looks odd, that may be why. Please let me know so I can fix it!

<figure>
  <a href="/images/ismir_2019_photo.jpeg"><img src="/images/ismir_2019_photo.jpeg" alt="Presenting our tutorial session at ISMIR 2019 in Delft, The Netherlands."></a>
  <figcaption>Presenting our tutorial session at ISMIR 2019 in Delft, The Netherlands. Via <a href="https://twitter.com/ismir2019/status/1191341227825934336">ISMIR2019 on Twitter</a>.</figcaption>
</figure>

## <a name="overview"></a> Overview

This blog post is divided into a few different sections. I'll try to motivate why modelling music in the waveform domain is an interesting problem. Then I'll give an overview of generative models, the various flavours that exist, and some important ways in which they differ from each other. In the next two sections I'll attempt to cover the state of the art in both likelihood-based and adversarial models of raw music audio. Finally, I'll raise some observations and discussion points. If you want to skip ahead, just click the section title below to go there.

* *[Motivation](#motivation)*
* *[Generative models](#generative-models)*
* *[Likelihood-based models of waveforms](#likelihood-based-models)*
* *[Adversarial models of waveforms](#adversarial-models)*
* *[Discussion](#discussion)*
* *[Conclusion](#conclusion)*
* *[References](#references)*

Note that this blog post is not intended to provide an exhaustive overview of all the published research in this domain -- I have tried to make a selection and I've inevitably left out some great work. **Please don't hesitate to suggest relevant work in the comments section!**


## <a name="motivation"></a> Motivation

### Why audio?

Music generation has traditionally been studied in the **symbolic domain**: the output of the generative process could be a musical score, a sequence of [MIDI events](https://en.wikipedia.org/wiki/MIDI), a simple melody, a sequence of chords, a textual representation[^folkrnn] or some other higher-level representation. The physical process through which sound is produced is abstracted away. This dramatically reduces the amount of information that the models are required to produce, which makes the modelling problem more tractable and allows for lower-capacity models to be used effectively.

A very popular representation is the so-called *piano roll*, which dates back to the player pianos of the early 20th century. Holes were punched into a roll of paper to indicate which notes should be played at which time. This representation survives in digital form today and is commonly used in music production. Much of the work on music generation using machine learning has made use of (some variant of) this representation, because it allows for capturing performance-specific aspects of the music without having to model the sound.


<figure class="half">
  <a href="/images/player_piano.jpg"><img src="/images/player_piano.jpg" alt="Player piano with a physical piano roll inside."></a>
  <a href="/images/piano_roll.jpg"><img src="/images/piano_roll.jpg" alt="Modern incarnation of a piano roll."></a>
  <figcaption><strong>Left:</strong> player piano with a physical piano roll inside. <strong>Right:</strong> modern incarnation of a piano roll.</figcaption>
</figure>

Piano rolls are great for piano performances, because they are able to exactly capture the *timing*, *pitch* and *velocity* (i.e. how hard a piano key is pressed, which is correlated with loudness, but not equivalent to it) of the notes. They are able to very accurately represent piano music, because they cover all the "degrees of freedom" that a performer has at their disposal. However, most other instruments have many more degrees of freedom: think about all the various ways you can play a note on the guitar, for example. You can decide which string to use, where to pick, whether to bend the string or not, play vibrato, ... you could even play harmonics, or use two-hand tapping. Such a vast array of different playing techniques endows the performer with a lot more freedom to vary the sound that the instrument produces, and coming up with a high-level representation that can accurately capture all this variety is much more challenging. In practice, a lot of this detail is ignored and a simpler representation is often used when generating music for these instruments.

Modelling the sound that an instrument produces is much more difficult than modelling (some of) the parameters that are controlled by the performer, but it frees us from having to manually design high-level representations that accurately capture all these parameters. Furthermore, it allows our models to capture variability that is beyond the performer's control: the idiosyncracies of individual instruments, for example (no two violins sound exactly the same!), or the parameters of the recording setup used to obtain the training data for our models. It also makes it possible to model ensembles of instruments, or other sound sources altogether, without having to fundamentally change anything about the model apart from the data it is trained on.

Digital audio representations require a reasonably high bit rate to achieve acceptable fidelity however, and modelling all these bits comes with a cost. **Music audio models will necessarily have to have a much higher capacity than their symbolic counterparts**, which implies higher computational requirements for model training.

### <a name="why-waveforms"></a>Why waveforms?

Digital representations of sound come in many shapes and forms. For reproduction, sound is usually stored by encoding the shape of the waveform as it changes over time. For analysis however, we often make use of **[spectrograms](https://en.wikipedia.org/wiki/Spectrogram)**, both for computational methods and for visual inspection by humans. A spectrogram can be obtained from a waveform by computing the Fourier transform of overlapping windows of the signal, and stacking the results into a 2D array. This shows the **local frequency content of the signal over time**.

Spectrograms are complex-valued: they represent both the amplitude and the phase of different frequency components at each point in time. Below is a visualisation of a magnitude spectrogram and its corresponding phase spectrogram. While the magnitude spectrogram clearly exhibits a lot of structure, with sustained frequencies manifesting as horizontal lines and harmonics showing up as parallel horizontal lines, the phase spectrogram looks a lot more random.

<figure>
  <a href="/images/spectrogram_magnitude.png"><img src="/images/spectrogram_magnitude.png" alt="Magnitude spectrogram of a piano recording."></a>
  <a href="/images/spectrogram_phase.png"><img src="/images/spectrogram_phase.png" alt="Phase spectrogram of a piano recording."></a>
  <figcaption><strong>Top:</strong> magnitude spectrogram of a piano recording. <strong>Bottom:</strong> the corresponding phase spectrogram.</figcaption>
</figure>

When extracting information from audio signals, it turns out that we can often just **discard the phase component**, because it is not informative for most of the things we could be interested in. In fact, this is why the magnitude spectrogram is often referred to simply as "the spectrogram". When generating sound however, phase is very important because it meaningfully affects our perception. Listen below to an original excerpt of a piano piece, and a corresponding excerpt where the original phase has been replaced by random uniform phase information. Note how the harmony is preserved, but the timbre changes completely.

<figure class="half">
    <audio controls src="/files/original_phase.wav"><a href="/files/original_phase.wav">Audio with original phase</a></audio>
    <audio controls src="/files/random_phase.wav"><a href="/files/random_phase.wav">Audio with random phase</a></audio>
    <figcaption><strong>Left:</strong> excerpt with original phase. <strong>Right:</strong> the same excerpt with random phase.</figcaption>
</figure>

The phase component of a spectrogram is tricky to model for a number of reasons:
- it is an **angle**: $$\phi \in [0, 2 \pi)$$ and it wraps around;
- it becomes **effectively random** as the magnitude tends towards 0, because noise starts to dominate;
- absolute phase is less meaningful, but **relative phase differences over time matter perceptually**.

If we model waveforms directly, we are implicitly modelling their phase as well, but we don't run into these issues that make modelling phase so cumbersome. There are other strategies to avoid these issues, some of which I will <a href="#alternatives">discuss later</a>, but **waveform modelling currently seems to be the dominant approach in the generative setting**. This is particularly interesting because magnitude spectrograms are by far the most common representation used for discriminative models of audio.

### Discretising waveforms

When representing a waveform digitally, we need to **discretise it in both time and amplitude**. This is referred to as [pulse code modulation (PCM)](https://en.wikipedia.org/wiki/Pulse-code_modulation). Because audio waveforms are effectively band-limited (humans cannot perceive frequencies above ~20 kHz), the [sampling theorem](https://en.wikipedia.org/wiki/Nyquist%E2%80%93Shannon_sampling_theorem) tells us that we can discretise the waveform in time without any loss of information, as long as the sample rate is high enough (twice the highest frequency). This is why CD quality audio has a sample rate of 44.1 kHz. Much lower sample rates result in an audible loss of fidelity, but since the resulting discrete sequences also end up being much shorter, a compromise is often struck in the context of generative modelling to reduce computational requirements. Most models from literature use sample rates of 16 or 24 kHz.

<figure>
  <a href="/images/digital_waveform.gif"><img style="width: 100%; border: 1px solid #eee;" src="/images/digital_waveform.gif" alt="Digital waveform."></a>
  <figcaption>Digital waveform. The individual samples become visible as the zoom level increases. Figure taken from <a href="https://deepmind.com/blog/article/wavenet-generative-model-raw-audio">the original WaveNet blog post</a>.</figcaption>
</figure>

When we also quantise the amplitude, some loss of fidelity is inevitable. CD quality uses 16 bits per sample, representing 2<sup>16</sup> equally spaced quantisation levels. If we want to use fewer bits, we can use logarithmically spaced quantisation levels instead to account for our nonlinear perception of loudness. This **["mu-law companding"](https://en.wikipedia.org/wiki/%CE%9C-law_algorithm)** will result in a smaller perceived loss of fidelity than if the levels were equally spaced.

## <a name="generative-models"></a> Generative models

Given a dataset $$X$$ of examples $$x \in X$$, which we assume to have been drawn independently from some underlying distribution $$p_X(x)$$, a generative model can learn to approximate this distribution $$p_X(x)$$. Such a model could be used to generate new samples that look like they could have been part of the original dataset. We distinguish *implicit* and *explicit* generative models: an implicit model can produce new samples $$x \sim p_X(x)$$, but cannot be used to infer the likelihood of an example (i.e. we cannot tractably compute $$p_X(x)$$ given $$x$$). If we have an explicit model, we can do this, though sometimes only up to an unknown normalising constant.

### Conditional generative models

Generative models become more practically useful when we can exert some influence over the samples we draw from them. We can do this by providing a **conditioning signal** $$c$$, which contains side information about the kind of samples we want to generate. The model is then fit to the conditional distribution $$p_X(x \vert c)$$ instead of $$p_X(x)$$.

Conditioning signals can take many shapes or forms, and it is useful to distinguish different levels of information content. The generative modelling problem becomes easier if the conditioning signal $$c$$ is richer, because it reduces uncertainty about $$x$$. We will refer to conditioning signals with low information content as *sparse conditioning*, and those with high information content as *dense conditioning*. Examples of conditioning signals in the image domain and the music audio domain are shown below, ordered according to density.

<figure>
  <img src="/images/sparse-dense-conditioning.svg" alt="Examples of sparse and dense conditioning signals in the image domain (top) and the music audio domain (bottom).">
  <figcaption>Examples of sparse and dense conditioning signals in the image domain (top) and the music audio domain (bottom).</figcaption>
</figure>

Note that the density of a conditioning signal is often correlated with its level of abstraction: high-level side information tends to be more sparse. Low-level side information isn't necessarily dense, though. For example, we could condition a generative model of music audio on a low-dimensional vector that captures the overall timbre of an instrument. This is a low-level aspect of the audio signal, but it constitutes a sparse conditioning signal.

### Likelihood-based models

Likelihood-based models directly parameterise $$p_X(x)$$. The parameters $$\theta$$ are then fit by maximising the likelihood of the data under the model:

$$\mathcal{L}_\theta(x) = \sum_{x \in X} \log p_X(x|\theta) \quad \quad \theta^* = \arg \max_\theta \mathcal{L}_\theta(x) .$$ 

Note that this is typically done in the log-domain because it simplifies computations and improves numerical stability. Because the model directly parameterises $$p_X(x)$$, we can **easily infer the likelihood of any** $$x$$, so we get an explicit model. Three popular flavours of likelihood-based models are autoregressive models, flow-based models and variational autoencoders. The following three subsections provide a brief overview of each.

### Autoregressive models

In an autoregressive model, we assume that our examples $$x \in X$$ can be treated as sequences $$\{x_i\}$$. We then factorise the distribution into a product of conditionals, using the [chain rule of probability](https://en.wikipedia.org/wiki/Chain_rule_(probability)):

$$p_X(x) = \prod_i p(x_i \vert x_{<i}) .$$

These conditional distributions are typically scalar-valued and much easier to model. Because we further assume that the distribution of the sequence elements is stationary, we can share parameters and use the same model for all the factors in this product.

For audio signals, this is a very natural thing to do, but we can also do this for other types of structured data by arbitrarily choosing an order (e.g. raster scan order for images, as in PixelRNN[^pixelrnn] and PixelCNN[^pixelcnn]).

Autoregressive models are attractive because they are able to **accurately capture correlations between the different elements** $$x_i$$ in a sequence, and they allow for fast inference (i.e. computing $$p_X(x)$$ given $$x$$). Unfortunately they tend to be **slow to sample from**, because samples need to be drawn sequentially from the conditionals for each position in the sequence.

### Flow-based models

Another strategy for constructing a likelihood-based model is to use the **[change of variables theorem](https://en.wikipedia.org/wiki/Probability_density_function#Function_of_random_variables_and_change_of_variables_in_the_probability_density_function)** to transform $$p_X(x)$$ into a simple, factorised distribution $$p_Z(z)$$ (standard Gaussian is a popular choice) using an invertible mapping $$x = g(z)$$:

$$p_X(x) = p_Z(z) \cdot |\det J|^{-1} \quad \quad J = \frac{dg(z)}{dz}.$$

Here, $$J$$ is the Jacobian of $$g(z)$$. Models that use this approach are referred to as normalising flows or flow-based models[^nice][^realnvp]. They are fast both for inference and sampling, but the **requirement for $$g(z)$$ to be invertible significantly constrains the model architecture**, and it makes them less parameter-efficient. In other words: flow-based models need to be quite large to be effective.

For an in-depth treatment of flow-based models, I recommend Eric Jang's [two-part blog post](https://blog.evjang.com/2018/01/nf1.html) on the subject, and [Papamakarios et al.'s excellent review paper](https://arxiv.org/abs/1912.02762).

### Variational autoencoders (VAEs)

By far the most popular class of likelihood-based generative models, I can't avoid mentioning variational[^vaerezende] autoencoders[^vaekingma] -- but **in the context of waveform modelling, they are probably the least popular approach**. In a VAE, we jointly learn two neural networks: an *inference network* $$q(z \vert x)$$ learns to probabilistically map examples $$x$$ into a latent space, and a *generative network* $$p(x \vert z)$$ learns the distribution of the data conditioned on a latent representation $$z$$. These are trained to maximise a lower bound on $$p_X(x)$$, called the ELBO (Evidence Lower BOund), because computing $$p_X(x)$$ given $$x$$ (exact inference) is not tractable.

Typical VAEs assume a factorised distribution for $$p(x \vert z)$$, which limits the extent to which they can capture dependencies in the data. While this is often an acceptable trade-off, in the case of waveform modelling it turns out to be a problematic restriction in practice. I believe this is why not a lot of work has been published that takes this approach (if you know of any, please point me to it). VAEs can also have more powerful decoders with fewer assumptions (autoregressive decoders, for example), but this may introduce other issues such as posterior collapse[^pc].

To learn more about VAEs, check out [Jaan Altosaar's tutorial](https://jaan.io/what-is-variational-autoencoder-vae-tutorial/).

### Adversarial models

Generative Adversarial Networks[^gans] (GANs) take a very different approach to capturing the data distribution. Two networks are trained simultaneously: a *generator* $$G$$ attempts to produce examples according to the data distribution $$p_X(x)$$, given latent vectors $$z$$, while a *discriminator* $$D$$ attempts to tell apart generated examples and real examples. In doing so, the discriminator provides a learning signal for the generator which enables it to better match the data distribution. In the original formulation, the loss function is as follows:

$$\mathcal{L}(x) = \mathbb{E}_x[\log D(x)] + \mathbb{E}_z[log(1 - D(G(z)))] .$$

The generator is trained to minimise this loss, whereas the discriminator attempts to maximise it. This means the training procedure is a **two-player minimax game**, rather than an optimisation process, as it is for most machine learning models. Balancing this game and keeping training stable has been one of the main challenges for this class of models. Many alternative formulations have been proposed to address this.

While adversarial and likelihood-based models are both ultimately trying to model $$p_X(x)$$, they approach this target from very different angles. As a result, **GANs tend to be better at producing realistic examples, but worse at capturing the full diversity of the data distribution**, compared to likelihood-based models.

### More exotic flavours

Many other strategies to learn models of complicated distributions have been proposed in literature. While research on waveform generation has chiefly focused on the two dominant paradigms of likelihood-based and adversarial models, some of these alternatives may hold promise in this area as well, so I want to mention a few that I've come across.

* **Energy-based models** measure the "energy" of examples, and are trained by fitting the model parameters so that examples coming from the dataset have low energy, whereas all other configurations of inputs have high energy. This amounts to fitting an unnormalised density.  A nice recent example is [the work by Du & Mordatch at OpenAI](https://openai.com/blog/energy-based-models/)[^energy]. Energy-based models have been around for a very long time though, and one could argue that likelihood-based models are a special case. 

* **Optimal transport** is another approach to measure the discrepancy between probability distributions, which has served as inspiration for new variants of generative adversarial networks[^wgan] and autoencoders[^swa].

* **Autoregressive implicit quantile networks**[^aiqn] use a similar network architecture as likelihood-based autoregressive models, but they are trained using the quantile regression loss, rather than maximimum likelihood.

* Two continuous distributions can be matched by minimising the L2 distance between the gradients of the density functions with respect to their inputs: $$\mathcal{L}(x) = \mathbb{E} [\vert\vert \nabla_x \log p_X(x) - \nabla_y \log p_Y(y) \vert\vert ^2]$$. This is called **score matching**[^scorematching] and some recent works have revisited this idea for density estimation[^ssm] and generative modelling[^scorebased].

* Please share any others that I haven't mentioned in the comments!

### Mode-covering vs. mode-seeking behaviour

An important consideration when determining which type of generative model is appropriate for a particular application, is the degree to which it is *mode-covering* or *mode-seeking*. When a model does not have enough capacity to capture all the variability in the data, different compromises can be made. If all examples should be reasonably likely under the model, it will have to overgeneralise and put probability mass on interpolations of examples that may not be meaningful (mode-covering). If there is no such requirement, the probability mass can be focused on a subset of examples, but then some parts of the distribution will be ignored by the model (mode-seeking).

<figure>
  <a href="/images/mode_seeking_covering.png"><img src="/images/mode_seeking_covering.png" alt="Illustration of mode-seeking and mode-covering behaviour in model fitting."></a>
  <figcaption>Illustration of mode-seeking and mode-covering behaviour in model fitting. The blue density represents the data distribution. The green density is our model, which is a single Gaussian. Because the data distribution is multimodal, our model does not have enough capacity to accurately capture it.</figcaption>
</figure>

**Likelihood-based models are usually mode-covering**. This is a consequence of the fact that they are fit by maximising the joint likelihood of the data. **Adversarial models on the other hand are typically mode-seeking**. A lot of ongoing research is focused on making it possible to control the trade-off between these two behaviours directly, without necessarily having to switch the class of models that are used.

In general, mode-covering behaviour is desirable in sparsely conditioned applications, where we want diversity or we expect a certain degree of "creativity" from the model. Mode-seeking behaviour is more useful in densely-conditioned settings, where most of the variability we care about is captured in the conditioning signal, and we favour realism of the generated output over diversity.

## <a name="likelihood-based-models"></a> Likelihood-based models of waveforms

In this section, I'll try to summarise some of the key results from the past four years obtained with likelihood-based models of waveforms. While this blog post is supposed to be about music, note that many of these developments were initially targeted at generating speech, so inevitably I will also be talking about some work in the text-to-speech (TTS) domain. I recommend reading the associated papers and/or blog posts to find out more about each of these works.

### WaveNet & SampleRNN

<figure>
  <a href="/images/wavenet.gif"><img style="display: block; margin: auto;" src="/images/wavenet.gif" alt="Wavenet sampling procedure."></a>
  <figcaption>Animation showing sampling from a WaveNet model. The model predicts the distribution of potential signal values for each timestep, given past signal values.</figcaption>
</figure>

WaveNet[^wavenet] and SampleRNN[^samplernn] are **autoregressive models of raw waveforms**. While WaveNet is a convolutional neural network, SampleRNN uses a stack of recurrent neural networks. Both papers appeared on arXiv in late 2016 with only a few months in between, signalling that autoregressive waveform-based audio modelling was an idea whose time had come. Before then, this idea had not been seriously considered, as modelling long-term correlations in sequences across thousands of timesteps did not seem feasible with the tools that were available at that point. Furthermore, discriminative models of audio all used spectral input representations, with only a few works investigating the use of raw waveforms in this setting (and usually with worse results).

Although these models have their flaws (including slow sampling due to autoregressivity, and a lack of interpretability w.r.t. what actually happens inside the network), I think they constituted an important *existence proof* that encouraged further research into waveform-based models.

WaveNet's strategy to deal with long-term correlations is to use *dilated convolutions*: successive convolutional layers use filters with gaps between their inputs, so that the connectivity pattern across many layers forms a tree structure (see figure above). This enables rapid growth of the receptive field, which means that **a WaveNet with only a few layers can learn dependencies across many timesteps**. Note that the convolutions used in WaveNet are causal (no connectivity from future to past), which forces the model to learn to predict what values the signal could take at each position in time.

SampleRNN's strategy is a bit different: multiple RNNs are stacked on top of each other, with each running at a different frequency. Higher-level RNNs update less frequently, which means they can more easily capture long-range correlations and learn high-level features.

Both models demonstrated excellent text-to-speech results, surpassing the state of the art at the time (concatenative synthesis, for most languages) in terms of naturalness. Both models were also applied to (piano) music generation, which constituted a nice demonstration of the promise of music generation in the waveform domain, but they were clearly limited in their ability to capture longer-term musical structure.

<p style='background-color: #efe; border: 1px dashed #898; padding: 0.2em 0.5em;'>
<strong>WaveNet</strong>: <a href="https://arxiv.org/abs/1609.03499">paper</a> - <a href="https://deepmind.com/blog/article/wavenet-generative-model-raw-audio">blog post</a><br>
<strong>SampleRNN</strong>: <a href="https://arxiv.org/abs/1612.07837">paper</a> - <a href="https://soundcloud.com/samplernn/sets">samples</a>
</p>

### Parallel WaveNet & ClariNet

Sampling from autoregressive models of raw audio can be quite slow and impractical. To address this issue, Parallel WaveNet[^parallelwavenet] uses *probability density distillation* to train a model from which samples can be drawn in a single feed-forward pass. This requires a trained autoregressive WaveNet, which functions as a teacher, and an inverse autoregressive flow (IAF) model which acts as the student and learns to mimic the teacher's predictions.

While an autoregressive model is slow to sample from, inferring the likelihood of a given example (and thus, maximum-likelihood training) can be done in parallel. **For an inverse autoregressive flow, it's the other way around: sampling is fast, but inference is slow**. Since most practical applications rely on sampling rather than inference, such a model is often better suited. IAFs are hard to train from scratch though (because that requires inference), and the probability density distillation approach makes training them tractable.

Due to the nature of the probability density distillation objective, the student will end up matching the teacher's predictions in a way that minimises the *reverse* KL divergence. This is quite unusual: likelihood-based models are typically trained to minimise the forward KL divergence instead, which is equivalent to maximising the likelihood (and minimising the reverse KL is usually intractable). While minimising the forward KL leads to mode-covering behaviour, **minimising the reverse KL will instead lead to mode-seeking behaviour**, which means that the model may end up ignoring certain modes in the data distribution.

In the text-to-speech (TTS) setting, this may actually be exactly what we want: given an excerpt of text, we want the model to generate a realistic utterance corresponding to that excerpt, but we aren't particularly fussed about being able to generate every possible variation -- one good-sounding utterance will do. This is a setting where **realism is clearly more important than diversity**, because all the diversity that we care about is already captured in the conditioning signal that we provide. This is usually the setting where adversarial models excel, because of their inherent mode-seeking behaviour, but using probability density distillation we can also train likelihood-based models this way.

To prevent the model from collapsing, parallel WaveNet uses a few additional loss terms to encourage the produced waveforms to resemble speech (such as a loss on the average power spectrum).

If we want to do music generation, we will typically care more about diversity because the conditioning signals we provide to the model are weaker. I believe this is why we haven't really seen the Parallel WaveNet approach catch on outside of TTS.

ClariNet[^clarinet] was introduced as a variant of Parallel WaveNet which uses a Gaussian inverse autoregressive flow. The Gaussian assumption makes it possible to compute the reverse KL in closed form, rather than having to approximate it by sampling, which stabilises training.

<p style='background-color: #efe; border: 1px dashed #898; padding: 0.2em 0.5em;'>
<strong>Parallel WaveNet</strong>: <a href="https://arxiv.org/abs/1711.10433">paper</a> - <a href="https://deepmind.com/blog/article/high-fidelity-speech-synthesis-wavenet">blog post 1</a> - <a href="https://deepmind.com/blog/article/wavenet-launches-google-assistant">blog post 2</a><br>
<strong>ClariNet</strong>: <a href="https://arxiv.org/abs/1807.07281">paper</a> - <a href="https://clarinet-demo.github.io/">samples</a>
</p>

### Flow-based models: WaveGlow, FloWaveNet, WaveFlow, Blow

Training an IAF with probability density distillation isn't the only way to train a flow-based model: most can be trained by maximum likelihood instead. In that case, the models will be encouraged to capture all the modes of the data distribution. This, in combination with their relatively low parameter efficiency (due to the invertibility requirement), means that they might need to be a bit larger to be effective. On the other hand, **they allow for very fast sampling because all timesteps can be generated in parallel**, so while the computational cost may be higher, sampling will still be faster in practice. Another advantage is that no additional loss terms are required to prevent collapse.

WaveGlow[^waveglow] and FloWaveNet[^flowavenet], both originally published in late 2018, are flow-based models of raw audio conditioned on mel-spectrograms, which means they can be used as *vocoders*. Because of the limited parameter efficiency of flow-based models, I suspect that it would be difficult to use them for music generation in the waveform domain, where conditioning signals are much more sparse -- but they could of course be used to render mel-spectrograms generated by some other model into waveforms (more on that later).

WaveFlow[^waveflow] (with an F instead of a G) is a more recent model that improves parameter efficiency by combining the flow-based modelling approach with partial autoregressivity to model local signal structure. This allows for a trade-off between sampling speed and model size. Blow[^blow] is a flow-based model of waveforms for non-parallel voice conversion. 

<p style='background-color: #efe; border: 1px dashed #898; padding: 0.2em 0.5em;'>
<strong>WaveGlow</strong>: <a href="https://arxiv.org/abs/1811.00002">paper</a> - <a href="https://github.com/NVIDIA/waveglow">code</a> - <a href="https://nv-adlr.github.io/WaveGlow">samples</a><br>
<strong>FloWaveNet</strong>: <a href="https://arxiv.org/abs/1811.02155">paper</a> - <a href="https://github.com/ksw0306/FloWaveNet">code</a> - <a href="https://ksw0306.github.io/flowavenet-demo/">samples</a><br>
<strong>WaveFlow</strong>: <a href="https://arxiv.org/abs/1912.01219">paper</a> - <a href="https://waveflow-demo.github.io/">samples</a><br>
<strong>Blow</strong>: <a href="https://papers.nips.cc/paper/8904-blow-a-single-scale-hyperconditioned-flow-for-non-parallel-raw-audio-voice-conversion">paper</a> - <a href="https://github.com/joansj/blow">code</a> - <a href="https://blowconversions.github.io/">samples</a>
</p>

### Hierarchical WaveNets

For the purpose of music generation, **WaveNet is limited by its ability to capture longer-term signal structure**, as previously stated. In other words: while it is clearly able to capture local signal structure very well (i.e. the timbre of an instrument), it isn't able to model the evolution of chord progressions and melodies over longer time periods. This makes the outputs produced by this model sound rather improvisational, to put it nicely.

This may seem counterintuitive at first: the tree structure of the connectivity between the layers of the model should allow for a very rapid growth of its receptive field. So if you have a WaveNet model that captures up to a second of audio at a time (more than sufficient for TTS), stacking a few more dilated convolutional layers on top should suffice to grow the receptive field by several orders of magnitude (up to many minutes). At that point, the model should be able to capture any kind of meaningful musical structure.

In practice, however, we need to train models on excerpts of audio that are at least as long as the longest-range correlations that we want to model. So while the depth of the model has to grow only logarithmically as we increase the desired receptive field, **the computational and memory requirements for training do in fact grow linearly**. If we want to train a model that can learn about musical structure across tens of seconds, that will necessarily be an order of magnitude more expensive -- and WaveNets that generate music already have to be quite large as it is, even with a receptive field of just one second, because **music is harder to model than speech**. Note also that one second of audio corresponds to a sequence of 16000 timesteps at 16 kHz, so even at a scale of seconds, we are already modelling very long sequences.

In 10 years, the hardware we would need to train a WaveNet with a receptive field of 30 seconds (or almost half a million timesteps at 16 kHz) may just fit in a desktop computer, so we could just wait until then to give it a try. But if we want to train such models today, we need a different strategy. If we could train separate models to capture structure at different timescales, we could have a dedicated model that focuses on capturing longer-range correlations, without having to also model local signal structure. This seems feasible, seeing as models of high-level representations of music (i.e. scores or MIDI) clearly do a much better job of capturing long-range musical structure already.

We can approach this as a **representation learning** problem: to decouple learning of local and large-scale structure, we need to extract a more compact, high-level representation $$h$$ from the audio signals $$x$$, that makes abstraction of local detail and has a much lower sample rate. Ideally, we would learn a model $$h = f(x)$$ to extract such a representation from data (although using existing high-level representations like MIDI is also possible, as we'll discuss later).

Then we can split up the task by training two separate models: a WaveNet that models the high-level representation: $$p_H(h)$$, and another that models the local signal structure, conditioned on the high-level representation: $$p_{X \vert H}(x \vert h)$$. The former model can focus on learning about long-range correlations, as local signal structure is not present in the representation it operates on. The latter model, on the other hand, can focus on learning about local signal structure, as relevant information about large-scale structure is readily available in its conditioning signal. Combined together, these models can be used to sample new audio signals by first sampling $$\hat{h} \sim p_H(h)$$ and then $$\hat{x} \sim p_{X \vert H}(x \vert \hat{h})$$.

We can learn both $$f(x)$$ and $$p_{X \vert H}(x \vert h)$$ together by training an *autoencoder*: $$f(x)$$ is the encoder, a feed-forward neural network, and $$p_{X \vert H}(x \vert h)$$ is the decoder, a conditional WaveNet. Learning these jointly will enable $$f(x)$$ to adapt to the WaveNet, so that it extracts information that the WaveNet cannot easily model itself.

To make the subsequent modelling of $$h = f(x)$$ with another WaveNet easier, we use a VQ-VAE[^vqvae]: an **autoencoder with a discrete bottleneck**. This has two important consequences:
- **Autoregressive models seem to be more effective on discrete sequences** than on continuous ones. Making the high-level representation discrete makes the hierarchical modelling task much easier, as we don't need to adapt the WaveNet model to work with continuous data.
- The discreteness of the representation also **limits its information capacity**, forcing the autoencoder to encode only the most important information in $$h$$, and to use the autoregressive connections in the WaveNet decoder to capture any local structure that wasn't encoded in $$h$$.

To split the task into more than two parts, we can apply this procedure again to the high-level representation $$h$$ produced by the first application, and **repeat this until we get a hierarchy with as many levels as desired**. Higher levels in the hierarchy make abstraction of more and more of the low-level details of the signal, and have progressively lower sample rates (yielding shorter sequences). a three-level hierarchy is shown in the diagram below. Note that **each level can be trained separately and in sequence**, thus greatly reducing the computational requirements of training a model with a very large receptive field.

<figure>
  <img src="/images/hierarchical_wavenet.svg" alt="Hierarchical WaveNet model, consisting of (conditional) autoregressive models of several levels of learnt discrete representations.">
  <figcaption>Hierarchical WaveNet model, consisting of (conditional) autoregressive models of several levels of learnt discrete representations.</figcaption>
</figure>

My colleagues and I explored this idea and trained hierachical WaveNet models on piano music[^challenge]. We found that there was a trade-off between audio fidelity and long-range coherence of the generated samples. When more model capacity was repurposed to focus on long-range correlations, this reduced the capability of the model to capture local structure, resulting in lower perceived audio quality. We also conducted a human evaluation study where we asked several listeners to rate both the fidelity and the musicality of some generated samples, to demonstrate that hierarchical models produce samples which sound more musical.

<p style='background-color: #efe; border: 1px dashed #898; padding: 0.2em 0.5em;'>
<strong>Hierarchical WaveNet</strong>: <a href="https://papers.nips.cc/paper/8023-the-challenge-of-realistic-music-generation-modelling-raw-audio-at-scale">paper</a> - <a href="https://drive.google.com/drive/folders/1s7yGi928cMla8gZhfQKNXACPACSrJ9Vg">samples</a>
</p>

### <a name="wave2midi2wave"></a> Wave2Midi2Wave and the MAESTRO dataset

As alluded to earlier, rather than learning high-level representations of music audio from data, we could also **use existing high-level representations such as MIDI** to construct a hierarchical model. We can use a powerful language model to model music in the symbolic domain, and also construct a conditional WaveNet model that generates audio, given a MIDI representation. Together with my colleagues from the Magenta team at Google AI, [we trained such models](https://magenta.tensorflow.org/maestro-wave2midi2wave) on a new dataset called MAESTRO, which features 172 hours of virtuosic piano performances, captured with fine alignment between note labels and audio waveforms[^maestro]. This dataset is [available to download](https://magenta.tensorflow.org/datasets/maestro) for research purposes.

Compared to hierarchical WaveNets with learnt intermediate representations, this approach yields much better samples in terms of musical structure, but it is limited to instruments and styles of music that MIDI can accurately represent. Manzelli et al. [have demonstrated this approach](http://people.bu.edu/bkulis/projects/music/index.html) for a few instruments other than piano[^manzellithakkar], but the lack of available aligned data could pose a problem.

<figure>
  <img src="/images/wave2midi2wave.png" alt="Wave2Midi2Wave: a transcription model to go from audio to MIDI, a transformer to model MIDI sequences and a WaveNet to synthesise audio given a MIDI sequence.">
  <figcaption>Wave2Midi2Wave: a transcription model to go from audio to MIDI, a transformer to model MIDI sequences and a WaveNet to synthesise audio given a MIDI sequence.</figcaption>
</figure>

<p style='background-color: #efe; border: 1px dashed #898; padding: 0.2em 0.5em;'>
<strong>Wave2Midi2Wave</strong>: <a href="https://openreview.net/forum?id=r1lYRjC9F7">paper</a> - <a href="https://magenta.tensorflow.org/maestro-wave2midi2wave">blog post</a> - <a href="https://storage.googleapis.com/magentadata/papers/maestro/index.html">samples</a> - <a href="https://magenta.tensorflow.org/datasets/maestro">dataset</a><br>
<strong>Manzelli et al. model</strong>: <a href="https://arxiv.org/abs/1806.09905">paper</a> - <a href="http://people.bu.edu/bkulis/projects/music/index.html">samples</a>
</p>

### Sparse transformers

OpenAI introduced the [Sparse Transformer](https://openai.com/blog/sparse-transformer/) model[^sparsetransformer], a large transformer[^transformer] with a **sparse attention mechanism** that scales better to long sequences than traditional attention (which is quadratic in the length of the modelled sequence). They demonstrated impressive results autoregressively modelling language, images, and music audio using this architecture, with sparse attention enabling their model to cope with waveforms of up to 65k timesteps (about 5 seconds at 12 kHz). The sparse attention mechanism seems like a good alternative to the stacked dilated convolutions of WaveNets, provided that an efficient implementation is available.

<p style='background-color: #efe; border: 1px dashed #898; padding: 0.2em 0.5em;'>
<strong>Sparse Transformer</strong>: <a href="https://arxiv.org/abs/1904.10509">paper</a> - <a href="https://openai.com/blog/sparse-transformer/">blog post</a> - <a href="https://soundcloud.com/openai_audio/sets/sparse_transformers">samples</a>
</p>

### Universal music translation network

An interesting conditional waveform modelling problem is that of "music translation" or "music style transfer": given a waveform, **render a new waveform where the same music is played by a different instrument**. The Universal Music Translation Network[^umtn] tackles this by training an autoencoder with multiple WaveNet decoders, where the encoded representation is encouraged to be agnostic to the instrument of the input (using an adversarial loss). A separate decoder is trained for each target instrument, so once this representation is extracted from a waveform, it can be synthesised in an instrument of choice. The separation is not perfect, but it works surprisingly well in practice. I think this is a nice example of a model that combines ideas from both likelihood-based models and the adversarial learning paradigm.

<p style='background-color: #efe; border: 1px dashed #898; padding: 0.2em 0.5em;'>
<strong>Universal music translation network</strong>: <a href="https://openreview.net/forum?id=HJGkisCcKm">paper</a> - <a href="https://github.com/facebookresearch/music-translation">code</a> - <a href="https://musictranslation.github.io/">samples</a>
</p>

### Dadabots

[Dadabots](http://dadabots.com) are a researcher / artist duo who have trained SampleRNN models on various albums (primarily metal) in order to produce more music in the same vein. These models aren't great at capturing long-range correlations, so it works best for artists whose style is naturally a bit disjointed. Below is a 24 hour livestream they've set up with a model generating infinite technical death metal in the style of 'Relentless Mutation' by Archspire.

<iframe width="560" height="315" src="https://www.youtube.com/embed/MwtVkPKx3RA" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## <a name="adversarial-models"></a> Adversarial models of waveforms

Adversarial modelling of audio has only recently started to see some successes, which is why this section is going to be a lot shorter than the previous one on likelihood-based models. The adversarial paradigm has been extremely successful in the image domain, but researchers have had a harder time translating that success to other domains and modalities, compared to likelihood-based models. As a result, published work so far has primarily focused on speech generation and the generation of individual notes or very short clips of music. As a field, we are still very much in the process of figuring out how to make GANs work well for audio at scale.

### WaveGAN

One of the first works to attempt using GANs for modelling raw audio signals is WaveGAN[^wavegan]. They trained a GAN on single-word speech recordings, bird vocalisations, individual drum hits and short excerpts of piano music. They also compared their raw audio-based model with a spectrogram-level model called SpecGAN. Although the fidelity of the [resulting samples](https://chrisdonahue.com/wavegan_examples/) is far from perfect in some cases, this work undoubtedly inspired a lot of researchers to take audio modelling with GANs more seriously.

<p style='background-color: #efe; border: 1px dashed #898; padding: 0.2em 0.5em;'>
<strong>WaveGAN</strong>: <a href="https://openreview.net/forum?id=ByMVTsR5KQ">paper</a> - <a href="https://github.com/chrisdonahue/wavegan">code</a> - <a href="https://chrisdonahue.com/wavegan_examples/">samples</a> - <a href="https://chrisdonahue.com/wavegan/">demo</a> - <a href="https://colab.research.google.com/drive/1e9o2NB2GDDjadptGr3rwQwTcw-IrFOnm">colab</a>
</p>

### GANSynth

So far in this blog post, we have focused on generating audio waveforms directly. However, I don't want to omit GANSynth[^gansynth], even though technically speaking it does not operate directly in the waveform domain. This is because the spectral representation it uses is **exactly invertible** -- no other models or phase reconstruction algorithms are used to turn the spectograms it generates into waveforms, which means it shares a lot of the advantages of models that operate directly in the waveform domain.

As <a href="#why-waveforms">discussed before</a>, modelling the phase component of a complex spectrogram is challenging, because the phase of real audio signals can seem essentially random. However, using some of its unique characteristics, we can transform the phase into a quantity that is easier to model and reason about: the *instantaneous frequency*. This is obtained by computing the temporal difference of the *unwrapped* phase between subsequent frames. "Unwrapping" means that we shift the phase component by a multiple of $$2 \pi$$ for each frame as needed to make it monotonic over time, as shown in the diagram below (because phase is an angle, all values modulo $$2 \pi$$ are equivalent).

**The instantaneous frequency captures how much the phase of a signal moves from one spectrogram frame to the next**. For harmonic sounds, this quantity is expected to be constant over time, as the phase rotates at a constant velocity. This makes this representation particularly suitable to model musical sounds, which have a lot of harmonic content (and in fact, it might also make the representation less suitable for modelling more general classes of audio signals, though I don't know if anyone has tried). For harmonic sounds, the instantaneous frequency is almost trivial to predict.

GANSynth is an adversarial model trained to produce the magnitude and instantaneous frequency spectrograms of recordings of individual musical notes. The trained model is also able to generalise to sequences of notes to some degree. [Check out the blog post](https://magenta.tensorflow.org/gansynth) for sound examples and more information.

<figure>
  <img src="/images/gansynth1.png" alt="Waveform with specrogram frame boundaries indicated as dotted lines.">
  <img src="/images/gansynth2.png" alt="From phase to instantaneous frequency.">
  <img src="/images/gansynth3.png" alt="Visualisations of the magnitude, phase, unwrapped phase and instantaneous frequency spectra of a real recording of a note.">
  <figcaption><strong>Top</strong>: waveform with specrogram frame boundaries indicated as dotted lines. <strong>Middle</strong>: from phase to instantaneous frequency. <strong>Bottom</strong>: visualisations of the magnitude, phase, unwrapped phase and instantaneous frequency spectra of a real recording of a note.</figcaption>
</figure>

<p style='background-color: #efe; border: 1px dashed #898; padding: 0.2em 0.5em;'>
<strong>GANSynth</strong>: <a href="https://openreview.net/forum?id=H1xQVn09FX">paper</a> - <a href="http://goo.gl/magenta/gansynth-code">code</a> - <a href="http://goo.gl/magenta/gansynth-examples">samples</a> - <a href="https://magenta.tensorflow.org/gansynth">blog post</a> - <a href="http://goo.gl/magenta/gansynth-demo">colab</a>
</p>

### <a name="melgan-gantts"></a> MelGAN & GAN-TTS

Two recent papers demonstrate excellent results using GANs for text-to-speech: MelGAN[^melgan] and GAN-TTS[^gantts]. The former also includes some music synthesis results, although fidelity is still an issue in that domain. The focus of MelGAN is inversion of magnitude spectrograms (potentially generated by other models), whereas as GAN-TTS is conditioned on the same "linguistic features" as the original WaveNet for TTS.

The architectures of both models share some interesting similarities, which shed light on the right inductive biases for raw waveform discriminators. Both models use **multiple discriminators at different scales**, each of which operates on a **random window** of audio extracted from the full sequence produced by the generator. This is similar to the patch-based discriminators that have occasionally been used in GANs for image generation. This windowing strategy seems to dramatically improve the capability of the generator to **correctly model high frequency content** in the audio signals, which is much more crucial to get right for audio than for images because it more strongly affects perceptual quality. The fact that both models benefited from this particular discriminator design indicates that we may be on the way to figuring out how to best design discriminator architectures for raw audio.

There are also some interesting differences: where GAN-TTS uses a combination of conditional and unconditional discriminators, MelGAN uses only unconditional discriminators and instead encourages the generator output to match the ground truth audio by adding an additional *feature matching* loss: the L1 distance between discriminator feature maps of real and generated audio. Both approaches seem to be effective.

Adversarial waveform synthesis is particularly useful for TTS, because it enables the use of highly parallelisable feed-forward models, which tend to have relatively low capacity requirements because they are trained with a mode-seeking loss. This means the models **can more easily be deployed on low-power hardware while still performing audio synthesis in real-time**, compared to autoregressive or flow-based models.

<p style='background-color: #efe; border: 1px dashed #898; padding: 0.2em 0.5em;'>
<strong>MelGAN</strong>: <a href="https://papers.nips.cc/paper/9629-melgan-generative-adversarial-networks-for-conditional-waveform-synthesis">paper</a> - <a href="https://github.com/descriptinc/melgan-neurips">code</a> - <a href="https://melgan-neurips.github.io/">samples</a><br>
<strong>GAN-TTS</strong>: <a href="https://openreview.net/forum?id=r1gfQgSFDr">paper</a> - <a href="https://github.com/mbinkowski/DeepSpeechDistances">code (FDSD)</a> - <a href="https://storage.googleapis.com/deepmind-media/research/abstract.wav">sample</a>
</p>

## <a name="discussion"></a> Discussion

To wrap up this blog post, I want to summarise a few thoughts about the current state of this area of research, and where things could be moving next.

### Why the emphasis on likelihood in music modelling?

Clearly, the dominant paradigm for generative models of music in the waveform domain is likelihood-based. This stands in stark contrast to the image domain, where adversarial approaches greatly outnumber likelihood-based ones. I suspect there are a few reasons for this (let me know if you think of any others):

* Compared to likelihood-based models, it seems like it has been harder to translate the successes of adversarial models in the image domain to other domains, and to the audio domain in particular. I think this is because in a GAN, the discriminator fulfills the role of a **domain-specific loss function**, and important prior knowledge that guides learning is encoded in its architecture. We have known about good architectural priors for images for a long time (stacks of convolutions), as evidenced by work on e.g. style transfer[^styletransfer] and the deep image prior[^deepimageprior]. For other modalities, we don't know as much yet. It seems we are now starting to figure out what kind of architectures work for waveforms (see <a href="#melgan-gantts">MelGAN and GAN-TTS</a>, some relevant work has also been done in the discriminative setting[^randomcnn]).

* **Adversarial losses are mode-seeking**, which makes them more suitable for settings where realism is more important than diversity (for example, because the conditioning signal contains most of the required diversity, as in TTS). In music generation, which is primarily a creative application, **diversity is very important**. Improving diversity of GAN samples is the subject of intense study right now, but I think it could be a while before they catch up with likelihood-based models in this sense.

* The current disparity could also simply be a consequence of the fact that **likelihood-based models got a head start** in waveform modelling, with WaveNet and SampleRNN appearing on the scene in 2016 and WaveGAN in 2018.

Another domain where likelihood-based models dominate is language modelling. I believe the underlying reasons for this might be a bit different though: language is inherently **discrete**, and extending GANs to modelling discrete data at scale is very much a work in progress. This is also more likely to be the reason why likelihood-based models are dominant for symbolic music generation as well: most symbolic representations of music are discrete.

### <a name="alternatives"></a> Alternatives to modelling waveforms directly

Instead of modelling music in the waveform domain, there are many possible alternative approaches. We could model other representations of audio signals, such as spectrograms, as long as we have a way to obtain waveforms from such representations. We have quite a few options for this:

* We could use **invertible spectrograms** (i.e. phase information is not discarded), but in this case modelling the phase poses a considerable challenge. There are ways to make this easier, such as the instantaneous frequency representation used by GANSynth.

* We could also use **magnitude spectrograms** (as is typically done in discriminative models of audio), and then use a **phase reconstruction algorithm** such as the Griffin-Lim algorithm[^griffinlim] to infer a plausible phase component, based only on the generated magnitude. This approach was used for the original Tacotron model for TTS[^tacotron], and for MelNet[^melnet], which models music audio autoregressively in the spectrogram domain.

* Instead of a traditional phase reconstruction algorithm, we could also use a **vocoder** to go from spectrograms to waveforms. A vocoder, in this context, is simply a generative model in the waveform domain, conditioned on spectrograms. Vocoding is a densely conditioned generation task, and many of the models discussed before can and have been used as vocoders (e.g. WaveNet in Tacotron 2[^tacotron2], flow-based models of waveforms, or MelGAN). This approach has some advantages: generated magnitude spectrograms are often imperfect, and vocoder models can learn to account for these imperfections. Vocoders can also work with inherently lossy spectrogram representations such as mel-spectrograms and constant-Q spectrograms[^constantq].
 
* If we are generating audio conditioned on an existing audio signal, we could also simply **reuse the phase** of the input signal, rather than reconstructing or generating it. This is commonly done in source separation, and the approach could also be used for music style transfer.

That said, modelling spectrograms **isn't always easier** than modelling waveforms. Although spectrograms have a much lower temporal resolution, they contain much more information per timestep. In autoregressive models of spectrograms, one would have to condition along both the time and frequency axes to capture all dependencies, which means we end up with roughly as many sequential sampling steps as in the raw waveform case. This is the approach taken by MelNet.

An alternative is to make an **assumption of independence between different frequency bands at each timestep**, given previous timesteps. This enables autoregressive models to produce entire spectrogram frames at a time. This partial independence assumption turns out to be an acceptable compromise in the text-to-speech domain, and is used in Tacotron and Tacotron 2. Vocoder models are particularly useful here as they can attempt to fix the imperfections resulting from this simplification of the model. I'm not sure if anybody has tried, but I would suspect that this independence assumption would cause more problems for music generation.

An interesting new approach combining traditional signal processing ideas with neural networks is [Differentiable Digital Signal Processing (DDSP)](https://magenta.tensorflow.org/ddsp)[^ddsp]. By creating learnable versions of existing DSP components and incorporating them directly into neural networks, these models are endowed with **much stronger inductive biases about sound and music**, and can learn to produce realistic audio with fewer trainable parameters, while also being more interpretable. I suspect that this research direction may gain a lot of traction in the near future, not in the least because the authors [have made their code publicly available](https://github.com/magenta/ddsp), and also because of its modularity and lower computational requirements.

<figure>
  <img src="/images/ddsp.png" alt="Diagram of an example DDSP model. The yellow boxes represent differentiable signal processing components.">
  <figcaption>Diagram of an example DDSP model. The yellow boxes represent differentiable signal processing components. Taken from <a href="https://magenta.tensorflow.org/ddsp">the original blog post</a>.</figcaption>
</figure>

Finally, we could train **symbolic models of music** instead: for many instruments, we already have realistic synthesisers, and we can even train them given enough data (see <a href="#wave2midi2wave">Wave2Midi2Wave</a>). If we are able to craft symbolic representations that capture the aspects of music we care about, then this is an attractive approach as it is much less computationally intensive. Magenta's [Music Transformer](https://magenta.tensorflow.org/music-transformer)[^musictransformer] and OpenAI's [MuseNet](https://openai.com/blog/musenet/) are two models that have recently shown impressive results in this domain, and it is likely that other ideas from the language modelling community could bring further improvements.

<p style='background-color: #efe; border: 1px dashed #898; padding: 0.2em 0.5em;'>
<strong>DDSP</strong>: <a href="https://openreview.net/forum?id=B1x1ma4tDr">paper</a> - <a href="https://github.com/magenta/ddsp">code</a> - <a href="https://g.co/magenta/ddsp-examples">samples</a> - <a href="https://magenta.tensorflow.org/ddsp">blog post</a> - <a href="https://g.co/magenta/ddsp-demo">colab</a><br>
<strong>Music Transformer</strong>: <a href="https://openreview.net/forum?id=rJe4ShAcF7">paper</a> - <a href="https://magenta.tensorflow.org/music-transformer">blog post</a><br>
<strong>MuseNet</strong>: <a href="https://openai.com/blog/musenet/">blog post</a>
</p>

### What's next?

Generative models of music in the waveform domain have seen substantial progress over the past few years, but the best results so far are still relatively easy to distinguish from real recordings, even at fairly short time scales. There is still a lot of room for improvement, but I believe a lot of this will be driven by better availability of computational resources, and not necessarily by radical innovation on the modelling front -- we have great tools already, they are simply a bit expensive to use due to **substantial computational requirements**. As time goes on and computers get faster, hopefully this task will garner interest as it becomes accessible to more researchers.

One interesting question is **whether adversarial models are going to catch up** with likelihood-based models in this domain. I think it is quite likely that GANs, having recently made in-roads in the densely conditioned setting, will gradually be made to work for more sparsely conditioned audio generation tasks as well.  Fully unconditional generation with long-term coherence seems very challenging however, and I suspect that the mode-seeking behaviour of the adversarial loss will make this much harder to achieve. A hybrid model, where a GAN captures local signal structure and another model with a different objective function captures high-level structure and long-term correlations, seems like a sensible thing to build.

**Hierarchy** is a very important prior for music (and, come to think of it, for pretty much anything else we like to model), so models that explicitly incorporate this are going to have a leg up on models that don't -- at the cost of some additional complexity. Whether this additional complexity will always be worth it remains to be seen, but at the moment, this definitely seems to be the case.

At any rate, **splitting up the problem into multiple stages** that can be solved separately has been fruitful, and I think it will continue to be. So far, hierarchical models (with learnt or handcrafted intermediate representations) and spectrogram-based models with vocoders have worked well, but perhaps there are other ways to "divide and conquer". A nice example of a different kind of split in the image domain is the one used in Subscale Pixel Networks[^spn], where separate networks model the most and least significant bits of the image data.

## <a name="conclusion"></a> Conclusion

If you made it to the end of this post, congratulations! I hope I've convinced you that music modelling in the waveform domain is an interesting research problem. It is also **very far from a solved problem**, so there are lots of opportunities for interesting new work. I have probably missed a lot of relevant references, especially when it comes to more recent work. If you know about relevant work that isn't discussed here, feel free to share it in the comments! Questions about this blog post and this line of research are very welcome as well.

<!-- TODO: add some bolded parts to highlight them where it makes sense. -->

## <a name="references"></a> References

[^folkrnn]: Sturm, Santos, Ben-Tal and Korshunova, "[Music transcription modelling and composition using deep learning](https://arxiv.org/pdf/1604.08723)", Proc. 1st Conf. Computer Simulation of Musical Creativity, Huddersfield, UK, July 2016. [folkrnn.org](https://folkrnn.org/)

[^pixelrnn]: Van den Oord, Kalchbrenner and Kavukcuoglu, "[Pixel recurrent neural networks](https://arxiv.org/abs/1601.06759)", International Conference on Machine Learning, 2016.

[^pixelcnn]: Van den Oord, Kalchbrenner, Espeholt, Vinyals and Graves, "[Conditional image generation with pixelcnn decoders](http://papers.nips.cc/paper/6527-conditional-image-generation-with-pixelcnn-decoders)", Advances in neural information processing systems 29 (NeurIPS), 2016.

[^nice]: Dinh, Krueger and Bengio, "[NICE: Non-linear Independent Components Estimation](https://arxiv.org/abs/1410.8516)", arXiv, 2014.

[^realnvp]: Dinh, Sohl-Dickstein and Bengio, "[Density estimation using Real NVP](https://arxiv.org/abs/1605.08803)", arXiv, 2016.

[^vaekingma]: Kingma and Welling, "[Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)", International Conference on Learning Representations, 2014.

[^vaerezende]: Rezende, Mohamed and Wierstra, "[Stochastic Backpropagation and Approximate Inference in Deep Generative Models](https://arxiv.org/abs/1401.4082)", International Conference on Machine Learning, 2014.

[^pc]: Bowman, Vilnis, Vinyals, Dai, Jozefowicz and Bengio, "[Generating Sentences from a Continuous Space](https://arxiv.org/abs/1511.06349)", 20th SIGNLL Conference on Computational Natural Language Learning, 2016.

[^gans]: Goodfellow, Pouget-Abadie, Mirza, Xu, Warde-Farley, Ozair, Courville and Bengio, "[Generative Adversarial Nets](http://papers.nips.cc/paper/5423-generative-adversarial-nets)", Advances in neural information processing systems 27 (NeurIPS), 2014.

[^aiqn]: Ostrovski, Dabney and Munos, "[Autoregressive Quantile Networks for Generative Modeling](https://arxiv.org/abs/1806.05575)", International Conference on Machine Learning, 2018.

[^scorematching]: Hyvärinen, "[Estimation of Non-Normalized Statistical Models by Score Matching](http://www.jmlr.org/papers/v6/hyvarinen05a.html)", Journal of Machine Learning Research, 2005.

[^energy]: Du and Mordatch, "[https://arxiv.org/abs/1903.08689](https://arxiv.org/abs/1903.08689)", arXiv, 2019.

[^wgan]: Arjovsky, Chintala and Bottou, "[Wasserstein GAN](https://arxiv.org/abs/1701.07875)", arXiv, 2017.

[^swa]: Kolouri, Pope, Martin and Rohde, "[Sliced-Wasserstein Autoencoder: An Embarrassingly Simple Generative Model](https://arxiv.org/abs/1804.01947)", arXiv, 2018.

[^ssm]: Song, Garg, Shi and Ermon, "[Sliced Score Matching: A Scalable Approach to Density and Score Estimation](https://arxiv.org/abs/1905.07088)", UAI, 2019.

[^scorebased]: Song and Ermon, "[Generative Modeling by Estimating Gradients of the Data Distribution](http://papers.nips.cc/paper/9361-generative-modeling-by-estimating-gradients-of-the-data-distribution)", Advances in neural information processing systems 32 (NeurIPS), 2019.

[^wavenet]: Van den Oord, Dieleman, Zen, Simonyan, Vinyals, Graves, Kalchbrenner, Senior and Kavukcuoglu, "[WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499)", arXiv, 2016.

[^samplernn]: Mehri, Kumar, Gulrajani, Kumar, Jain, Sotelo, Courville and Bengio, "[SampleRNN: An Unconditional End-to-End Neural Audio Generation Model](https://arxiv.org/abs/1612.07837)", International Conference on Learning Representations, 2017.

[^parallelwavenet]: Van den Oord, Li, Babuschkin, Simonyan, Vinyals, Kavukcuoglu, van den Driessche, Lockhart, Cobo, Stimberg, Casagrande, Grewe, Noury, Dieleman, Elsen, Kalchbrenner, Zen, Graves, King, Walters, Belov and Hassabis, "[Parallel WaveNet: Fast High-Fidelity Speech Synthesis](https://arxiv.org/abs/1711.10433)", International Conference on Machine Learning, 2018.

[^clarinet]: Ping, Peng and Chen, "[ClariNet: Parallel Wave Generation in End-to-End Text-to-Speech](https://arxiv.org/abs/1807.07281)", International Conference on Learning Representations, 2019.

[^waveglow]: Prenger, Valle and Catanzaro, "[WaveGlow: A Flow-based Generative Network for Speech Synthesis](https://arxiv.org/abs/1811.00002)", International Conference on Acoustics, Speech, and Signal Procesing, 2019

[^flowavenet]: Kim, Lee, Song, Kim and Yoon, "[FloWaveNet : A Generative Flow for Raw Audio](https://arxiv.org/abs/1811.02155)", International Conference on Machine Learning, 2019.

[^waveflow]: Ping, Peng, Zhao and Song, "[WaveFlow: A Compact Flow-based Model for Raw Audio](https://arxiv.org/abs/1912.01219)", ArXiv, 2019.

[^blow]: Serrà, Pascual and Segura, "[Blow: a single-scale hyperconditioned flow for non-parallel raw-audio voice conversion](https://papers.nips.cc/paper/8904-blow-a-single-scale-hyperconditioned-flow-for-non-parallel-raw-audio-voice-conversion)", Advances in neural information processing systems 32 (NeurIPS), 2019.

[^vqvae]: Van den Oord, Vinyals and Kavukcuoglu, "[Neural Discrete Representation Learning](http://papers.nips.cc/paper/7210-neural-discrete-representation-learning)", Advances in neural information processing systems 30 (NeurIPS), 2017.

[^challenge]: Dieleman, Van den Oord and Simonyan, "[The challenge of realistic music generation: modelling raw audio at scale](https://papers.nips.cc/paper/8023-the-challenge-of-realistic-music-generation-modelling-raw-audio-at-scale)", Advances in neural information processing systems 31 (NeurIPS), 2018.

[^maestro]: Hawthorne, Stasyuk, Roberts, Simon, Huang, Dieleman, Elsen, Engel and Eck, "[Enabling Factorized Piano Music Modeling and Generation with the MAESTRO Dataset](https://openreview.net/forum?id=r1lYRjC9F7)", International Conference on Learning Representations, 2019.

[^manzellithakkar]: Manzelli, Thakkar, Siahkamari and Kulis, "[Conditioning Deep Generative Raw Audio Models for Structured Automatic Music](https://arxiv.org/abs/1806.09905)", International Society for Music Information Retrieval Conference, 2018.

[^sparsetransformer]: Child, Gray, Radford and Sutskever, "[Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509)", Arxiv, 2019.

[^transformer]: Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser and Polosukhin, "[Attention is All you Need](http://papers.nips.cc/paper/7181-attention-is-all-you-need)", Advances in neural information processing systems 30 (NeurIPS), 2017.

[^umtn]: Mor, Wolf, Polyak and Taigman, "[A Universal Music Translation Network](https://openreview.net/forum?id=HJGkisCcKm)", International Conference on Learning Representations, 2019.

[^wavegan]: Donahue, McAuley and Puckette, "[Adversarial Audio Synthesis](https://openreview.net/forum?id=ByMVTsR5KQ)", International Conference on Learning Representations, 2019.

[^gansynth]: Engel, Agrawal, Chen, Gulrajani, Donahue and Roberts, "[GANSynth: Adversarial Neural Audio Synthesis](https://openreview.net/forum?id=H1xQVn09FX)", International Conference on Learning Representations, 2019.

[^melgan]: Kumar, Kumar, de Boissiere, Gestin, Teoh, Sotelo, de Brébisson, Bengio and Courville, "[MelGAN: Generative Adversarial Networks for Conditional Waveform Synthesis](https://papers.nips.cc/paper/9629-melgan-generative-adversarial-networks-for-conditional-waveform-synthesis)", Advances in neural information processing systems 32 (NeurIPS), 2019.

[^gantts]: Bińkowski, Donahue, Dieleman, Clark, Elsen, Casagrande, Cobo and Simonyan, "[High Fidelity Speech Synthesis with Adversarial Networks](https://openreview.net/forum?id=r1gfQgSFDr)", International Conference on Learning Representations, 2020.

[^styletransfer]: Gatys, Ecker and Bethge, "[Image Style Transfer Using Convolutional Neural Networks](http://openaccess.thecvf.com/content_cvpr_2016/html/Gatys_Image_Style_Transfer_CVPR_2016_paper.html)", IEEE Conference on Computer Vision and Pattern Recognition, 2016.

[^deepimageprior]: Ulyanov, Vedaldi and Lempitsky, "[Deep Image Prior](http://openaccess.thecvf.com/content_cvpr_2018/html/Ulyanov_Deep_Image_Prior_CVPR_2018_paper.html)", IEEE Conference on Computer Vision and Pattern Recognition, 2018.

[^randomcnn]: Pons and Serra, "[Randomly weighted CNNs for (music) audio classification](https://arxiv.org/abs/1805.00237)", IEEE International Conference on Acoustics, Speech and Signal Processing, 2019.

[^griffinlim]: Griffin and Lim, "[Signal estimation from modified short-time Fourier transform](https://ieeexplore.ieee.org/abstract/document/1164317/)", IEEE Transactions on Acoustics, Speech and Signal Processing, 1984.

[^tacotron]: Wang, Skerry-Ryan, Stanton, Wu, Weiss, Jaitly, Yang, Xiao, Chen, Bengio, Le, Agiomyrgiannakis, Clark and Saurous, "[Tacotron: Towards end-to-end speech synthesis](https://arxiv.org/abs/1703.10135)", Interspeech, 2017.

[^melnet]: Vasquez and Lewis, "[Melnet: A generative model for audio in the frequency domain](https://arxiv.org/abs/1906.01083)", ArXiv, 2019.

[^tacotron2]: Shen, Pang, Weiss, Schuster, Jaitly, Yang, Chen, Zhang, Wang, Skerry-Ryan, Saurous, Agiomyrgiannakis, Wu, "[Natural TTS synthesis by conditioning wavenet on mel spectrogram predictions](https://arxiv.org/abs/1712.05884)", IEEE International Conference on Acoustics, Speech and Signal Processing, 2018.

[^constantq]: Schörkhuber and Klapuri, "[Constant-Q transform toolbox for music processing](https://iem.kug.ac.at/fileadmin/media/iem/projects/2010/smc10_schoerkhuber.pdf)", Sound and Music Computing Conference, 2010.

[^ddsp]: Engel, Hantrakul, Gu and Roberts, "[DDSP: Differentiable Digital Signal Processing](https://openreview.net/forum?id=B1x1ma4tDr)", International Conference on Learning Representations, 2020.

[^musictransformer]: Huang, Vaswani, Uszkoreit, Simon, Hawthorne, Shazeer, Dai, Hoffman, Dinculescu and Eck, "[Music Transformer: Generating Music with Long-Term Structure ](https://openreview.net/forum?id=rJe4ShAcF7)", International Conference on Learning Representations, 2019.

[^spn]: Menick and Kalchbrenner, "[Generating High Fidelity Images with Subscale Pixel Networks and Multidimensional Upscaling](https://openreview.net/forum?id=HylzTiC5Km)", International Conference on Learning Representations, 2019.