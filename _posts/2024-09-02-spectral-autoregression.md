---
layout: post
title: "Diffusion is spectral autoregression"
description: "A deep dive into spectral analysis of diffusion models of images, revealing how they implicitly perform a form of autoregression in the frequency domain."

tags: [diffusion, autoregression, spectrum, spectral analysis, Fourier transform, natural images, deep learning, generative models]

image:
  feature: rainbow3.jpg
comments: true
share: true
---

A bit of signal processing swiftly reveals that diffusion models and autoregressive models aren't all that different: **diffusion models of images perform approximate autoregression in the frequency domain!**

<p style='background-color: #eee; padding: 1.2em; font-weight: bold; margin: 2em 0; text-align: center;'>
This blog post is also available as a <a href="https://colab.research.google.com/drive/1siywvhvl1OxI1UmqRrJHiFUK0M5SHlcx">Python notebook in Google Colab <img src="/images/colab_logo.png" style="height: 1.5em; margin-left: 0.2em; vertical-align: middle;"></a>, with the code used to produce all the plots and animations.</p>


Last year, I wrote a blog post describing various different [perspectives on diffusion](https://sander.ai/2023/07/20/perspectives.html). The idea was to highlight a number of connections between diffusion models and other classes of models and concepts. In recent months, I have given a few talks where I discussed some of these perspectives. My talk at the [EEML 2024 summer school](https://www.eeml.eu/) in Novi Sad, Serbia, was recorded and is [available on YouTube](https://www.youtube.com/watch?v=9BHQvQlsVdE). Based on the response I got from this talk, the link between diffusion models and **autoregressive models** seems to be particularly thought-provoking. That's why I figured it could be useful to explore this a bit further. 

In this blog post, I will unpack the above claim, and try to make it obvious that this is the case, at least for visual data. To make things more tangible, I decided to write this entire blog post in the form of [a Python notebook](https://colab.research.google.com/drive/1siywvhvl1OxI1UmqRrJHiFUK0M5SHlcx) (using Google Colab). That way, **you can easily reproduce the plots and analyses yourself**, and modify them to observe what happens. I hope this format will also help drive home the point that this connection between diffusion models and autoregressive models is "real", and not just a theoretical idealisation that doesn't hold up in practice.

In what follows, I will assume a basic understanding of diffusion models and the core concepts behind them. If you've watched the talk I linked above, you should be able to follow along. Alternatively, the [perspectives on diffusion](https://sander.ai/2023/07/20/perspectives.html) blog post should also suffice as preparatory reading. Some knowledge of the Fourier transform will also be helpful.

Below is an overview of the different sections of this post. Click to jump directly to a particular section.

1. *[Two forms of iterative refinement](#iterative-refinement)*
2. *[A spectral view of diffusion](#spectral-view)*
3. *[What about sound?](#sound)*
4. *[Unstable equilibrium](#unstable-equilibrium)*
5. *[Closing thoughts](#closing-thoughts)*
6. *[Acknowledgements](#acknowledgements)*
7. *[References](#references)*

## <a name="iterative-refinement"></a> Two forms of iterative refinement

<figure>
  <a href="/images/jonction.jpg"><img src="/images/jonction.jpg"></a>
</figure>

Autoregression and diffusion are currently the two dominant generative modelling paradigms. There are many more ways to build generative models: [flow-based models](https://en.wikipedia.org/wiki/Flow-based_generative_model) and [adversarial models](https://en.wikipedia.org/wiki/Generative_adversarial_network) are just two possible alternatives (I discussed a few more in [an earlier blog post](https://sander.ai/2020/03/24/audio-generation.html#generative-models)).

Both autoregression and diffusion differ from most of these alternatives, by splitting up the difficult task of generating data from complex distributions into smaller subtasks that are easier to learn. Autoregression does this by casting the data to be modelled into the shape of a sequence, and recursively predicting one sequence element at a time. Diffusion instead works by defining a corruption process that gradually destroys all structure in the data, and training a model to learn to invert this process step by step.

This **iterative refinement** approach to generative modelling is very powerful, because it allows us to construct very deep computational graphs for generation, without having to backpropagate through them during training. Indeed, both autoregressive models and diffusion models learn to perform a single step of refinement at a time -- the generative process is not trained end-to-end. It is only when we try to sample from the model that we connect all these steps together, by sequentially performing the subtasks: predicting one sequence element after another in the case of autoregression, or gradually denoising the input step-by-step in the case of diffusion.

Because this underlying iterative approach is common to both paradigms, people have often sought to connect the two. One could frame autoregression as a special case of discrete diffusion, for example, with a corruption process that gradually replaces tokens by "mask tokens" from right to left, eventually ending up with a fully masked sequence. In the next few sections, we will do the opposite, framing diffusion as a special case of autoregression, albeit approximate.

Today, most language models are autoregressive, while most models of images and video are diffusion-based. In many other application domains (e.g. protein design, planning in reinforcement learning, ...), diffusion models are also becoming more prevalent. I think this dichotomy, which can be summarised as "autoregression for language, and diffusion for everything else", is quite interesting. I have [written about it before](https://sander.ai/2023/01/09/diffusion-language.html), and I will have more to say about it in a later section of this post.


## <a name="spectral-view"></a> A spectral view of diffusion

<figure>
  <a href="/images/prism.jpg"><img src="/images/prism.jpg"></a>
</figure>

### <a name="image-spectra"></a> Image spectra

When diffusion models rose to prominence for image generation, people noticed quite quickly that they tend to produce images in a coarse-to-fine manner. The large-scale structure present in the image seems to be decided in earlier denoising steps, whereas later denoising steps add more and more fine-grained details.

To formalise this observation, we can use signal processing, and more specifically **spectral analysis**. By decomposing an image into its constituent **spatial frequency** components, we can more precisely tease apart its coarse- and fine-grained structure, which correspond to low and high frequencies respectively.

We can use the 2D [Fourier transform](https://en.wikipedia.org/wiki/Fourier_transform) to obtain a frequency representation of an image. This representation is invertible, i.e. it contains the same information as the pixel representation -- it is just organised in a different way. Like the pixel representation, it is a 2D grid-structured object, with the same width and height as the original image, but the axes now correspond to horizontal and vertical spatial frequencies, rather than spatial positions.

To see what this looks like, let's take some images and visualise their spectra.

<figure style="text-align: center;">
  <a href="/images/plot_image_spectra.png"><img src="/images/plot_image_spectra.png" alt="Four images from the Imagenette dataset (top), along with their magnitude spectra (middle) and their phase spectra (bottom)."></a>
  <figcaption>Four images from the <a href="https://github.com/fastai/imagenette">Imagenette dataset</a> (top), along with their magnitude spectra (middle) and their phase spectra (bottom).</figcaption>
</figure>

Shown above on the first row are four images from the [Imagenette dataset](https://github.com/fastai/imagenette), a subset of the ImageNet dataset (I picked it because it is relatively fast to load).

The Fourier transform is typically complex-valued, so the next two rows visualise the _magnitude_ and the _phase_ of the spectrum respectively. Because the magnitude varies greatly across different frequencies, its logarithm is shown. The phase is an angle, which varies between $$-\pi$$ and $$\pi$$. Note that we only calculate the spectrum for the green colour channel -- we could calculate it for the other two channels as well, but they would look very similar.

The centre of the spectrum corresponds to the lowest spatial frequencies, and the frequencies increase as we move outward to the edges. This allows us to see where most of the energy in the input signal is concentrated. Note that by default, it is the other way around (low frequencies in the corner, high frequencies in the middle), but `np.fft.fftshift` allows us to swap these, which yields a much nicer looking visualisation that makes the structure of the spectrum more apparent.

A lot of interesting things can be said about the phase structure of natural images, but in what follows, we will primarily focus on the magnitude spectrum. The square of the magnitude is the _power_, so in practice we often look at the _power spectrum_ instead. Note that the logarithm of the power spectrum is simply that of the magnitude spectrum, multiplied by two.

Looking at the spectra, we now have a more formal way to reason about different feature scales in images, but that still doesn't explain why diffusion models exhibit this coarse-to-fine behaviour. To see why this happens, we need to examine what a typical image spectrum looks like. To do this, we will **make abstraction of the directional nature of frequencies in 2D space**, simply by slicing the spectrum along a certain angle, rotating that slice all around, and then averaging the slices across all rotations. This yields a one-dimensional curve: **the _radially averaged power spectral density_, or RAPSD**.

Below is an animation that shows individual directional slices of the 2D spectrum on a log-log plot, which are averaged to obtain the RAPSD.

<figure style="text-align: center;">
  <a href="/images/image_spectrum.gif"><img src="/images/image_spectrum.gif" alt="Animation that shows individual directional slices of the 2D spectrum of an image on a log-log plot."></a>
  <figcaption>Animation that shows individual directional slices of the 2D spectrum of an image on a log-log plot.</figcaption>
</figure>

Let's see what that looks like for the four images above. We will use the `pysteps` library, which comes with a handy function to calculate the RAPSD in one go.

<figure style="text-align: center;">
  <a href="/images/plot_image_rapsd.png"><img src="/images/plot_image_rapsd.png" alt="Four images from the Imagenette dataset (top), along with their radially averaged spectral power densities (RAPSDs, bottom)."></a>
  <figcaption>Four images from the Imagenette dataset (top), along with their radially averaged spectral power densities (RAPSDs, bottom).</figcaption>
</figure>

The RAPSD is best visualised on a log-log plot, to account for the large variation in scale. We chop off the so-called DC component (with frequency 0) to avoid taking the logarithm of 0.

Another thing this visualisation makes apparent is that the curves are remarkably close to being straight lines. A straight line on a log-log plot implies that there might be a power law lurking behind all of this.

Indeed, this turns out to be the case: **natural image spectra tend to approximately follow a power law**, which means that the power $$P(f)$$ of a particular frequency $$f$$ is proportional to $$f^{-\alpha}$$, where $$\alpha$$ is a parameter[^schaaf] [^torralba] [^hyvarinen]. In practice, $$\alpha$$ is often remarkably close to 2 (which corresponds to the spectrum of [pink noise](https://en.wikipedia.org/wiki/Pink_noise) in two dimensions).

We can get closer to the "typical" RAPSD by taking the average across a bunch of images (in the log-domain).

<figure style="text-align: center;">
  <a href="/images/mean_log_rapsd.png"><img src="/images/mean_log_rapsd.png" alt="The average of RAPSDs of a set of images in the log-domain."></a>
  <figcaption>The average of RAPSDs of a set of images in the log-domain.</figcaption>
</figure>

As I'm sure you will agree, that is pretty unequivocally a power law!

To estimate the exponent $$\alpha$$, we can simply use linear regression in log-log space. Before proceeding however, it is useful to resample our averaged RAPSD so the sample points are linearly spaced in log-log space -- otherwise our fit will be dominated by the high frequencies, where we have many more sample points.

We obtain an estimate $$\hat{\alpha} = 2.454$$, which is a bit higher than the typical value of 2. As far as I understand, this can be explained by the presence of man-made objects in many of the images we used, because they tend to have smooth surfaces and straight angles, which results in comparatively more low-frequency content and less high-frequency content compared to images of nature. Let's see what our fit looks like.

<figure style="text-align: center;">
  <a href="/images/mean_log_rapsd_fit.png"><img src="/images/mean_log_rapsd_fit.png" alt="The average of RAPSDs of a set of images in the log-domain (red line), along with a linear fit (dotted black line)."></a>
  <figcaption>The average of RAPSDs of a set of images in the log-domain (red line), along with a linear fit (dotted black line).</figcaption>
</figure>

### <a name="noisy-spectra"></a> Noisy image spectra

A crucial aspect of diffusion models is the corruption process, which involves adding Gaussian noise. Let's see what this does to the spectrum. The first question to ask is: what does the spectrum of noise look like? We can repeat the previous procedure, but replace the image input with standard Gaussian noise. For contrast, we will visualise the spectrum of the noise alongside that of the images from before.

<figure style="text-align: center;">
  <a href="/images/mean_log_rapsd_noise.png"><img src="/images/mean_log_rapsd_noise.png" alt="The average of RAPSDs of a set of images in the log-domain (red line), along with the average of RAPSDs of standard Gaussian noise (blue line)."></a>
  <figcaption>The average of RAPSDs of a set of images in the log-domain (red line), along with the average of RAPSDs of standard Gaussian noise (blue line).</figcaption>
</figure>

The RAPSD of Gaussian noise is also a straight line on a log-log plot; but a horizontal one, rather than one that slopes down. This reflects the fact that **Gaussian noise contains all frequencies in equal measure**. The Fourier transform of Gaussian noise is itself Gaussian noise, so its power must be equal across all frequencies in expectation.

When we add noise to the images and look at the spectrum of the resulting noisy images, we see a hinge shape:

<figure style="text-align: center;">
  <a href="/images/mean_log_rapsd_sum.png"><img src="/images/mean_log_rapsd_sum.png" alt="The average of RAPSDs of a set of images in the log-domain (red line), along with the average of RAPSDs of standard Gaussian noise (blue line) and the average of RAPSDs of their sum (green line)."></a>
  <figcaption>The average of RAPSDs of a set of images in the log-domain (red line), along with the average of RAPSDs of standard Gaussian noise (blue line) and the average of RAPSDs of their sum (green line).</figcaption>
</figure>

Why does this happen? Recall that the **Fourier transform is linear**: the Fourier transform of the sum of two things, is the sum of the Fourier transforms of those things. Because the power of the different frequencies varies across orders of magnitude, **one of the terms in this sum tends to drown out the other**. This is what happens at low frequencies, where the image spectrum dominates, and hence the green curve overlaps with the red curve. At high frequencies on the other hand, the noise spectrum dominates, and the green curve overlaps with the blue curve. In between, there is a transition zone where the power of both spectra is roughly matched.

If we increase the variance of the noise by scaling the noise term, we increase its power, and as a result, its RAPSD will shift upward (which is also a consequence of the linearity of the Fourier transform). This means a smaller part of the image spectrum now juts out above the waterline: **the increasing power of the noise looks like the rising tide!**

<figure style="text-align: center;">
  <a href="/images/mean_log_rapsd_high_noise.png"><img src="/images/mean_log_rapsd_high_noise.png" alt="The average of RAPSDs of a set of images in the log-domain (red line), along with the average of RAPSDs of Gaussian noise with variance 16 (blue line) and the average of RAPSDs of their sum (green line)."></a>
  <figcaption>The average of RAPSDs of a set of images in the log-domain (red line), along with the average of RAPSDs of Gaussian noise with variance 16 (blue line) and the average of RAPSDs of their sum (green line).</figcaption>
</figure>


At this point, I'd like to revisit a diagram from the [perspectives on diffusion blog post](https://sander.ai/2023/07/20/perspectives.html#autoregressive), where I originally drew the connection between diffusion and autoregression in frequency space, which is shown below.

<figure style="text-align: center;">
  <a href="/images/image_spectra.png"><img src="/images/image_spectra.png" alt="Magnitude spectra of natural images, Gaussian noise, and noisy images."></a>
  <figcaption>Magnitude spectra of natural images, Gaussian noise, and noisy images.</figcaption>
</figure>

These idealised plots of the spectra of images, noise, and their superposition match up pretty well with the real versions. When I originally drew this, I didn't actually realise just how closely this reflects reality!

What these plots reveal is an approximate equivalence (in expectation) between adding noise to images, and **low-pass filtering** them. The noise will drown out some portion of the high frequencies, and leave the low frequencies untouched. The variance of the noise determines the **cut-off frequency** of the filter. Note that this is the case only because of the characteristic shape of natural image spectra.

The animation below shows how the spectrum changes as we gradually add more noise, until it eventually overpowers all frequency components, and all image content is gone.

<figure style="text-align: center;">
  <a href="/images/rising_tide.gif"><img src="/images/rising_tide.gif" alt="Animation that shows the changing averaged RAPSD as more and more noise is added to a set of images."></a>
  <figcaption>Animation that shows the changing averaged RAPSD as more and more noise is added to a set of images.</figcaption>
</figure>

### <a name="diffuion"></a> Diffusion

With this in mind, it becomes apparent that the corruption process used in diffusion models is actually gradually filtering out more and more high-frequency information from the input image, and the different time steps of the process correspond to a frequency decomposition: basically an approximate version of the **Fourier transform**!

Since diffusion models themselves are tasked with reversing this corruption process step-by-step, they end up roughly predicting the next higher frequency component at each step of the generative process, given all preceding (lower) frequency components. This is a soft version of **autoregression in frequency space**, or if you want to make it sound fancier, **appproximate spectral autoregression**.

To the best of my knowledge, [Rissanen et al. (2022)](https://arxiv.org/abs/2206.13397)[^heat] were the first to apply this kind of analysis to diffusion in the context of generative modelling (see §2.2 in the paper). Their work directly inspired this blog post.

In many popular formulations of diffusion, the corruption process does not just involve adding noise, but also rescaling the input to keep the total variance within a reasonable range (or constant, in the case of variance-preserving diffusion). I have largely ignored this so far, because it doesn't materially change anything about the intuitive interpretation. Scaling the input simply results in the RAPSD shifting up or down a bit.


### <a name="quantitative"></a> Which frequencies are modelled at which noise levels?

There seems to be a monotonic relationship between noise levels and spatial frequencies (and hence feature scales). Can we characterise this quantitatively?

We can try, but it is important to emphasise that this relationship is only really valid in expectation, averaged across many images: **for individual images, the spectrum will not be a perfectly straight line, and it will not typically be monotonically decreasing**.

Even if we ignore all that, the "elbow" of the hinge-shaped spectrum of a noisy image is not very sharp, so it is clear that there is quite a large transition zone where we cannot unequivocally say that a particular frequency is dominated by either signal or noise. So this is, at best, a very smooth approximation to the "hard" autoregression used in e.g. large language models.

Keeping all of that in mind, let us construct a mapping from noise levels to frequencies for a particular diffusion process and a particular image distribution, by choosing a signal-to-noise ratio (SNR) threshold, below which we will consider the signal to be undetectable. This choice is quite arbitrary, and we will just have to choose a value and stick with it. We can choose 1 to keep things simple, which means that we consider the signal to be detectable if its power is equal to or greater than the power of the noise.

Consider a Gaussian diffusion process for which $$\mathbf{x}_t = \alpha(t)\mathbf{x}_0 + \sigma(t) \mathbf{\varepsilon}$$, with $$\mathbf{x}_0$$ an example from the data distribution, and $$\mathbf{\varepsilon}$$ standard Gaussian noise.

Let us define $$\mathcal{R}[\mathbf{x}](f)$$ as the RAPSD of an image $$\mathbf{x}$$ evaluated at frequency $$f$$. We will call the SNR threshold $$\tau$$. If we consider a particular time step $$t$$, then assuming the RAPSD is monotonically decreasing, we can define the **maximal detectable frequency** $$f_\max$$ at this time step in the process as the maximal value of $$f$$ for which:

$$ \mathcal{R}[\alpha(t)\mathbf{x}_0](f) > \tau \cdot \mathcal{R}[\sigma(t)\mathbf{\varepsilon}](f). $$

Recall that the Fourier transform is a linear operator, and $$\mathcal{R}$$ is a radial average of the square of its magnitude. Therefore, scaling the input to $$\mathcal{R}$$ by a real value means the output gets scaled by its square. We can use this to simplify things:

$$ \mathcal{R}[\mathbf{x}_0](f) > \tau \cdot \frac{\sigma(t)^2}{\alpha(t)^2} \mathcal{R}[\mathbf{\varepsilon}](f). $$

We can further simplify this by noting that $$\forall f: \mathcal{R}[\mathbf{\varepsilon}](f) = 1$$:

$$ \mathcal{R}[\mathbf{x}_0](f) > \tau \cdot \frac{\sigma(t)^2}{\alpha(t)^2}. $$

To construct such a mapping in practice, we first have to choose a diffusion process, which gives us the functional form of $$\sigma(t)$$ and $$\alpha(t)$$. To keep things simple, we can use the rectified flow[^rectifiedflow] / flow matching[^flowmatching] process, as used in Stable Diffusion 3[^sd3], for which $$\sigma(t) = t$$ and $$\alpha(t) = 1 - t$$. Combined with $$\tau = 1$$, this yields:

$$ \mathcal{R}[\mathbf{x}_0](f) > \left(\frac{t}{1 - t}\right)^2. $$

With these choices, we can now determine the shape of $$f_\max(t)$$ and visualise it.

<figure style="text-align: center;">
  <a href="/images/max_detectable_frequency.png"><img src="/images/max_detectable_frequency.png" alt="Maximum detectable frequency as a function of diffusion time, for a given set of images and the diffusion process used in rectified flow and flow matching formalisms."></a>
  <figcaption>Maximum detectable frequency as a function of diffusion time, for a given set of images and the diffusion process used in rectified flow and flow matching formalisms.</figcaption>
</figure>

The frequencies here are relative: if the bandwidth of the signal is 1, then 0.5 corresponds to the [Nyquist frequency](https://en.wikipedia.org/wiki/Nyquist_frequency), i.e. the maximal frequency that is representable with the given bandwidth.

Note that all representable frequencies are detectable at time steps near 0. As $$t$$ increases, so does the noise level, and hence $$f_\max$$ starts dropping, until it eventually reaches 0 (no detectable signal frequencies are left) close to $$t = 1$$.

## <a name="sound"></a> What about sound?

<figure>
  <a href="/images/mixer.jpg"><img src="/images/mixer.jpg"></a>
</figure>

All of the analysis above hinges on the fact that spectra of natural images typically follow a power law. Diffusion models have also been used to generate audio[^wavegrad] [^diffwave], which is the other main perceptual modality besides the visual. A very natural question to ask is whether the same interpretation makes sense in the audio domain as well.

To establish that, we will grab a dataset of typical audio recordings that we might want to build a generative model of: speech and music.

<figure style="text-align: center;">
    <audio controls src="/files/audio_clip1.wav"><a href="/files/audio_clip1.wav">Audio clip 1</a></audio>
    <a href="/images/spectrogram_clip1.png"><img src="/images/spectrogram_clip1.png" alt="Magnitude spectrogram for audio clip 1."></a>
    <audio controls src="/files/audio_clip2.wav"><a href="/files/audio_clip2.wav">Audio clip 2</a></audio>
    <a href="/images/spectrogram_clip2.png"><img src="/images/spectrogram_clip2.png" alt="Magnitude spectrogram for audio clip 2."></a>
    <audio controls src="/files/audio_clip3.wav"><a href="/files/audio_clip3.wav">Audio clip 3</a></audio>
    <a href="/images/spectrogram_clip3.png"><img src="/images/spectrogram_clip3.png" alt="Magnitude spectrogram for audio clip 3."></a>
    <audio controls src="/files/audio_clip4.wav"><a href="/files/audio_clip4.wav">Audio clip 4</a></audio>
    <a href="/images/spectrogram_clip4.png"><img src="/images/spectrogram_clip4.png" alt="Magnitude spectrogram for audio clip 4."></a>
    <figcaption>Four audio clips from the <a href="https://www.kaggle.com/datasets/lnicalo/gtzan-musicspeech-collection">GTZAN music/speech dataset</a>, and their corresponding spectrograms.</figcaption>
</figure>

Along with each audio player, a _spectrogram_ is shown: this is a time-frequency representation of the sound, which is obtained by applying the Fourier transform to short overlapping windows of the waveform and stacking the resulting magnitude vectors together in a 2D matrix.

For the purpose of comparing the spectrum of sound with that of images, we will use the 1-dimensional analogue of the RAPSD, which is simply the squared magnitude of the 1D Fourier transform.

<figure style="text-align: center;">
  <a href="/images/plot_sound_spectra.png"><img src="/images/plot_sound_spectra.png" alt="Magnitude spectra of four audio clips from the GTZAN music/speech dataset."></a>
  <figcaption>Magnitude spectra of four audio clips from the <a href="https://www.kaggle.com/datasets/lnicalo/gtzan-musicspeech-collection">GTZAN music/speech dataset</a>.</figcaption>
</figure>

These are a lot noisier than the image spectra, which is not surprising as these are not averaged over directions, like the RAPSD is. But aside from that, they don't really look like straight lines either -- the power law shape is nowhere to be seen!

I won't speculate about why images exhibit this behaviour and sound seemingly doesn't, but it is certainly interesting (feel free to speculate away in the comments!). To get a cleaner view, we can again average the spectra of many clips in the log domain, as we did with the RAPSDs of images.

<figure style="text-align: center;">
  <a href="/images/mean_log_spec.png"><img src="/images/mean_log_spec.png" alt="The average of magnitude spectra of a set of audio clips the log-domain."></a>
  <figcaption>The average of magnitude spectra of a set of audio clips the log-domain.</figcaption>
</figure>

Definitely not a power law. More importantly, it is not monotonic, so adding progressively more Gaussian noise to this does not obfuscate frequencies in descending order: **the "diffusion is just spectral autoregression" meme does not apply to audio waveforms!**

The average spectrum of our dataset exhibits a peak around 300-400 Hz. This is not too far off the typical spectrum of [green noise](https://en.wikipedia.org/wiki/Colors_of_noise#Green_noise), which has more energy in the region of 500 Hz. Green noise is supposed to sound like "the background noise of the world".

<figure style="text-align: center;">
  <a href="/images/rising_tide_sound.gif"><img src="/images/rising_tide_sound.gif" alt="Animation that shows the changing averaged magnitude spectrum as more and more noise is added to a set of audio clips."></a>
  <figcaption>Animation that shows the changing averaged magnitude spectrum as more and more noise is added to a set of audio clips.</figcaption>
</figure>

As the animation above shows, the different frequencies present in audio signals still get filtered out gradually from least powerful to most powerful, because the spectrum of Gaussian noise is still flat, just like in the image domain. But as the audio spectrum does not monotonically decay with increasing frequency, the order is not monotonic in terms of the frequencies themselves.

What does this mean for diffusion in the waveform domain? That's not entirely clear to me. It certainly makes the link with autoregressive models weaker, but I'm not sure if there are any negative implications for generative modelling performance.

One observation that does perhaps indicate that this is the case, is that a lot of diffusion models of audio described in the literature **do not operate directly in the waveform domain**. It is quite common to first extract some form of spectrogram (as we did earlier), and perform diffusion in that space, essentially treating it like an image[^hawthorne] [^riffusion] [^edmsound]. Note that spectrograms are a somewhat lossy representation of sound, because [phase information is typically discarded](https://sander.ai/2020/03/24/audio-generation.html#why-waveforms).

To understand the implications of this for diffusion models, we will extract **log-scaled mel-spectrograms** from the sound clips we have used before. The [mel scale](https://en.wikipedia.org/wiki/Mel_scale) is a nonlinear frequency scale which is intended to be perceptually uniform, and which is very commonly used in spectral analysis of sound.

Next, we will interpret these spectrograms as images and look at their spectra. Taking the spectrum of a spectrum might seem odd -- some of you might even suggest that it is pointless, because the Fourier transform is its own inverse! But note that there are a few nonlinear operations happening in between: taking the magnitude (discarding the phase information), mel-binning and log-scaling. As a result, this second Fourier transform doesn't just undo the first one.

<figure style="text-align: center;">
  <a href="/images/plot_sound_melspec_rapsd.png"><img src="/images/plot_sound_melspec_rapsd.png" alt="RAPSDs of mel-spectrograms of four audio clips from the GTZAN music/speech dataset."></a>
  <figcaption>RAPSDs of mel-spectrograms of four audio clips from the <a href="https://www.kaggle.com/datasets/lnicalo/gtzan-musicspeech-collection">GTZAN music/speech dataset</a>.</figcaption>
</figure>

It seems like the power law has resurfaced! We can look at the average in the log-domain again to get a smoother curve.

<figure style="text-align: center;">
  <a href="/images/mean_log_melspec_rapsd.png"><img src="/images/mean_log_melspec_rapsd.png" alt="The average of RAPSDs of mel-spectrograms of a set of sound clips in the log-domain (red line), along with a linear fit (dotted black line)."></a>
  <figcaption>The average of RAPSDs of mel-spectrograms of a set of sound clips in the log-domain (red line), along with a linear fit (dotted black line).</figcaption>
</figure>

I found this pretty surprising. I actually used to object quite strongly to the idea of treating spectrograms as images, as in this tweet in response to [Riffusion](https://en.wikipedia.org/wiki/Riffusion), a variant of Stable Diffusion finetuned on spectrograms:

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Me: &quot;NOOO, you can&#39;t just treat spectrograms as images, the frequency and time axes have completely different semantics, there is no locality in frequency and ...&quot;<br><br>These guys: &quot;Stable diffusion go brrr&quot; <a href="https://t.co/Akv8aZl8Rv">https://t.co/Akv8aZl8Rv</a></p>&mdash; Sander Dieleman (@sedielem) <a href="https://twitter.com/sedielem/status/1603412454427574279?ref_src=twsrc%5Etfw">December 15, 2022</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

... but I have always had to concede that it seems to work pretty well in practice, and perhaps the fact that spectrograms exhibit power-law spectra is one reason why.

There is also an interesting link with [mel-frequency cepstral coefficients (MFCCs)](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum), a popular feature representation for speech and music processing which predates the advent of deep learning. These features are constructed by taking the [discrete cosine transform (DCT)](https://en.wikipedia.org/wiki/Discrete_cosine_transform) of a mel-spectrogram. The resulting spectrum-of-a-spectrum is often referred to as the **cepstrum**.

So with this approach, perhaps the meme applies to sound after all, albeit with a slight adjustment: **diffusion on spectrograms is just cepstral autoregression**.

## <a name="unstable-equilibrium"></a> Unstable equilibrium

<figure>
  <a href="/images/spinningtop.jpg"><img src="/images/spinningtop.jpg"></a>
</figure>

So far, we have talked about a spectral perspective on diffusion, but we have not really discussed how it can be used to explain why diffusion works so well for images. The fact that this interpretation is possible for images, but not for some other domains, does not automatically imply that the method should also work better.

However, it does mean that the diffusion loss, which is a weighted average across all noise levels, is also implicitly a weighted average over all spatial frequencies in the image domain. Being able to individually weight these frequencies in the loss according to their relative importance is key, because the sensitivity of the human visual system to particular frequencies varies greatly. **This effectively makes the diffusion training objective a kind of perceptual loss**, and I believe it largely explains the success of diffusion models in the visual domain (together with [classifier-free guidance](https://sander.ai/2023/08/28/geometry.html)).

Going beyond images, one could use the same line of reasoning to try and understand why diffusion models _haven't_ really caught on in the domain of language modelling so far (I wrote more about this [last year](https://sander.ai/2023/01/09/diffusion-language.html)). The interpretation in terms of a frequency decomposition is not really applicable there, and hence being able to change the relative weighting of noise levels in the loss doesn't quite have the same impact on the quality of generated outputs.

For language modelling, autoregression is currently the dominant modelling paradigm, and while diffusion-based approaches have been making inroads recently[^ratios] [^sahoo] [^shi], a full-on takeover does not look like it is in the cards in the short term.

This results in the following status quo: **we use autoregression for language, and we use diffusion for pretty much everything else**. Of course, I realise that I have just been arguing that these two approaches are not all that different in spirit. But in practice, their implementations can look quite different, and a lot of knowledge and experience that practitioners have built up is specific to each paradigm.

To me, this feels like an **unstable equilibrium, because the future is multimodal**. We will ultimately want models that natively understand language, images, sound and other modalities mixed together. Grafting these two different modelling paradigms together to construct multimodal models is effective to some extent, and certainly interesting from a research perspective, but it brings with it an increased level of complexity (i.e. having to master two different modelling paradigms) which I don't believe practitioners will tolerate in the long run.

So in the longer term, it seems plausible that we could go back to using autoregression across all modalities, perhaps borrowing some ideas from diffusion in the process[^var] [^arnovq]. Alternatively, we might figure out how to build multimodal diffusion models for all modalities, including language. I don't know which it is going to be, but both of those outcomes ultimately seem more likely than the current situation persisting.

One might ask, if diffusion is really just approximate autoregression in frequency space, why not just do exact autoregression in frequency space instead, and maybe that will work just as well? That would mean we can use autoregression across all modalities, and resolve the "instability" in one go. [Nash et al. (2021)](https://arxiv.org/abs/2103.03841)[^dctransformer], [Tian et al. (2024)](https://arxiv.org/abs/2404.02905)[^var] and [Mattar et al. (2024)](https://arxiv.org/abs/2406.19997)[^wavelets] explore this direction.

There is a good reason not to take this shortcut, however: the diffusion sampling procedure is exceptionally flexible, in ways that autoregressive sampling is not. For example, the number of sampling steps can be chosen at test time (this isn't impossible for autoregressive models, but it is much less straightforward to achieve). This flexibility also enables [various distillation methods](https://sander.ai/2024/02/28/paradox.html) to reduce the number of steps required, and [classifier-free guidance](https://sander.ai/2023/08/28/geometry.html) to improve sample quality. Before we do anything rash and ditch diffusion altogether, we will probably want to figure out a way to avoid having to give up some of these benefits.

## <a name="closing-thoughts"></a> Closing thoughts

<figure>
  <a href="/images/lake_sunset.jpg"><img src="/images/lake_sunset.jpg"></a>
</figure>


When I first had a closer look at the spectra of real images myself, I realised that the link between diffusion models and autoregressive models is even stronger than I had originally thought -- in the image domain, at least. This is ultimately why I decided to write this blog post in [a notebook](https://colab.research.google.com/drive/1siywvhvl1OxI1UmqRrJHiFUK0M5SHlcx), to make it easier for others to see this for themselves as well. More broadly speaking, I find that learning by "doing" has a much more lasting effect than learning by reading, and hopefully making this post interactive can help with that.

There are of course many other ways to connect the two modelling paradigms of diffusion and autoregression, which I won't go into here, but it is becoming a rather popular topic of inquiry[^rolling] [^fifo] [^forcing].

If you enjoyed this post, I strongly recommend also reading [Rissanen et al. (2022)](https://arxiv.org/abs/2206.13397)'s paper on generative modelling with inverse heat dissipation[^heat], which inspired it.

This blog-post-in-a-notebook was an experiment, so any feedback on the format is very welcome! It's a bit more work, but hopefully some readers will derive some benefit from it. If there are enough of you, perhaps I will do more of these in the future. **Please share your thoughts in the comments!**

To wrap up, below are some low-effort memes I made when I should have been working on this blog post instead.

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">The interpretation of diffusion as autoregression in the frequency domain seems to be stirring up a lot of thought! (I may or may not have a new blog post in the works 🧐) <a href="https://t.co/XSxP27pKSt">pic.twitter.com/XSxP27pKSt</a></p>&mdash; Sander Dieleman (@sedielem) <a href="https://twitter.com/sedielem/status/1820233922287919263?ref_src=twsrc%5Etfw">August 4, 2024</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet"><p lang="en" dir="ltr">It&#39;s so much easier to tweet low-effort memes which assert that diffusion is just autoregression in frequency space, than it is to write a blog post about it 🤷 (but I&#39;m doing both!) <a href="https://t.co/snLQavtZBf">pic.twitter.com/snLQavtZBf</a></p>&mdash; Sander Dieleman (@sedielem) <a href="https://twitter.com/sedielem/status/1826728256542052800?ref_src=twsrc%5Etfw">August 22, 2024</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

<br><br>

*If you would like to cite this post in an academic context, you can use this BibTeX snippet:*

```
@misc{dieleman2024spectral,
  author = {Dieleman, Sander},
  title = {Diffusion is spectral autoregression},
  url = {https://sander.ai/2024/09/02/spectral-autoregression.html},
  year = {2024}
}
```

## <a name="acknowledgements"></a> Acknowledgements

Thanks to my colleagues at Google DeepMind for various discussions, which continue to shape my thoughts on this topic! In particular, thanks to Robert Riachi, Ruben Villegas and Daniel Zoran.

## <a name="references"></a> References

[^schaaf]: van der Schaaf, van Hateren, "[Modelling the Power Spectra of Natural Images: Statistics and Information](https://www.sciencedirect.com/science/article/pii/0042698996000028)", Vision Research, 1996.

[^torralba]: Torralba, Oliva, "[Statistics of natural image categories](https://web.mit.edu/torralba/www/ne3302.pdf)", Network: Computation in Neural Systems, 2003.

[^hyvarinen]: Hyvärinen, Hurri, Hoyer, "[Natural Image Statistics: A probabilistic approach to early computational vision](https://dl.acm.org/doi/abs/10.5555/1572513)", 2009. 

[^rectifiedflow]: Liu, Gong, Liu, "[Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003)", International Conference on Learning Representations, 2023.

[^flowmatching]: Lipman, Chen, Ben-Hamu, Nickel, Le, "[Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)", International Conference on Learning Representations, 2023.

[^sd3]: Esser, Kulal, Blattmann, Entezari, Muller, Saini, Levi, Lorenz, Sauer, Boesel, Podell, Dockhorn, English, Lacey, Goodwin, Marek, Rombach, "[Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206)", arXiv, 2024.

[^wavegrad]: Chen, Zhang, Zen, Weiss, Norouzi, Chan, "[WaveGrad: Estimating Gradients for Waveform Generation](https://arxiv.org/abs/2009.00713)", International Conference on Learning Representations, 2021.

[^diffwave]: Kong, Ping, Huang, Zhao, Catanzaro, "[DiffWave: A Versatile Diffusion Model for Audio Synthesis](https://arxiv.org/abs/2009.09761)", International Conference on Learning Representations, 2021.

[^heat]: Rissanen, Heinonen, Solin, "[Generative Modelling With Inverse Heat Dissipation](https://arxiv.org/abs/2206.13397)", International Conference on Learning Representations, 2023.

[^hawthorne]: Hawthorne, Simon, Roberts, Zeghidour, Gardner, Manilow, Engel, "[Multi-instrument Music Synthesis with Spectrogram Diffusion](https://arxiv.org/abs/2206.05408)", International Society for Music Information Retrieval conference, 2022.

[^riffusion]: Forsgren, Martiros, "[Riffusion](https://en.wikipedia.org/wiki/Riffusion)", 2022.

[^edmsound]: Zhu, Wen, Carbonneau, Duan, "[EDMSound: Spectrogram Based Diffusion Models for Efficient and High-Quality Audio Synthesis](https://arxiv.org/abs/2311.08667)", Neural Information Processing Systems Workshop on Machine Learning for Audio, 2023.

[^ratios]: Lou, Meng, Ermon, "[Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution](https://arxiv.org/abs/2310.16834)", International Conference on Machine Learning, 2024.

[^sahoo]: Sahoo, Arriola, Schiff, Gokaslan, Marroquin, Chiu, Rush, Kuleshov, "[Simple and Effective Masked Diffusion Language Models](https://arxiv.org/abs/2406.07524)", arXiv, 2024.

[^shi]: Shi, Han, Wang, Doucet, Titsias, "[Simplified and Generalized Masked Diffusion for Discrete Data](https://arxiv.org/abs/2406.04329)", arXiv, 2024.

[^var]: Tian, Jiang, Yuan, Peng, Wang, "[Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction](https://arxiv.org/abs/2404.02905)", arXiv, 2024. 

[^arnovq]: Li, Tian, Li, Deng, He, "[Autoregressive Image Generation without Vector Quantization](https://arxiv.org/abs/2406.11838)", arXiv, 2024.

[^dctransformer]: Nash, Menick, Dieleman, Battaglia, "[Generating Images with Sparse Representations](https://arxiv.org/abs/2103.03841)", International Conference on Machine Learning, 2021.

[^wavelets]: Mattar, Levy, Sharon, Dekel, "[Wavelets Are All You Need for Autoregressive Image Generation](https://arxiv.org/abs/2406.19997)", arXiv, 2024.

[^rolling]: Ruhe, Heek, Salimans, Hoogeboom, "[Rolling Diffusion Models](https://arxiv.org/abs/2402.09470)", International Conference on Machine Learning, 2024.

[^fifo]: Kim, Kang, Choi, Han, "[FIFO-Diffusion: Generating Infinite Videos from Text without Training](https://arxiv.org/abs/2405.11473)", arXiv, 2024.

[^forcing]: Chen, Monso, Du, Simchowitz, Tedrake, Sitzmann, "[Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion](https://arxiv.org/abs/2407.01392)", arXiv, 2024.
