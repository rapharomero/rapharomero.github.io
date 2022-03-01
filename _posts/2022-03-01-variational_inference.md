---
layout: post
title: Variational Inference
author:
  - Raphaël Romero

header-includes:
  - \usepackage{mathbbm}
bibliography:
  - "bibtex.bib"
date: March 1 2022
---

In this post, we give an overview of variational methods for Bayesian inference.

<!--more-->

# Introduction

We consider a setup where we observe a random variable $x\in\mathcal{X}$, that is conditioned by a unobserved variable $z \in \mathcal{Z}$. Examples of such setups include Latent Dirichlet allocation or Latent space models for graphs.

# Bayesian Setting

We adopt a Bayesian view, where we provide a prior distribution $p(z)$ on the hidden variable $z$.

We would like to perform inference on the variable $z$ conditioned on the observation of $x$, namely our goal is to find a posterior distribution $p(z\vert x)$.

Using Baye's rule, the latter is given by :

$$p(z\vert x) = \frac{p(x\vert z) p(z)}{p(x)}$$

## The evidence and its intractability

Evaluating the posterior above involved evaluating the denominator, also called the _evidence_ :

$$p(x) = \int\limits p(x\vert z) p(z) dz$$

This evaluation requires integrating over a high dimensional latent space. In some cases the integrand $p(x\vert z) p(z)$ might adopt a nice form, making the integral tractable possibly in closed form. However in the general case computing this high dimensional integral is difficult.

## A possible approach: Monte-Carlo Markov Chain

In order to tackle the intractability of the evidence, a traditional method involves approximating this integral by sampling from a Markov Chain, and using the obtain samples ($z_1,...,z_n$) to compute a Monte Carlo estimate of the form:

$$p(x) \approx \frac{1}{n}\sum\limits_{i=1}^{n} p(x\vert z_i) p(z_i) $$

In the most common approaches (for instance the Metropolis-Hastings algorithm), the Markov transitions only require evaluating the numerator $p(x \vert z_i)p(z_i)$, and under some hypotheses the Markov chain is guaranteed to cover the latent space after a certain number of interations.

While this approach allows to estimate the exact posterior distribution, it suffers from the curse of dimensionality, since the number of samples requires to get a good Monte-Carlo estimate scales exponentially with the latent space dimension.

# Variational inference

In order to counter the effects of dimensionality, another different approach is to estimate an approximation of the posterior.

As we will see, such an approximation casts the inference problem into an optimization problem, where the optimization variable is a density function. The term _variational_ comes from the fact we use a function, $q$ as an optimization variable in that formulation.

## Jensen's inequality and the ELBO

This is done by using Jensen's inequality: for any positive density $z \mapsto q(z)$ we have:

$$
\begin{aligned}\log(p(x)) &= log(\int\limits \frac{p(x, z)}{q(z)} q(z) dz)\\ &\geq
\int\limits \log(\frac{p(x, z)}{q(z)}) q(z)) dz) \\ &=
F(x, q)
\end{aligned}
$$

where we define the functional $$q \mapsto F(x, q) = \int\limits \log(\frac{p(x, z)}{q(z)}) q(z)) dz).$$ $F$ is commonly known as the $ELBO$ in variational inference litterature. As we can see, it is a function of both the observation $x$ and the density $q$.

## Link with the Kullback-Leibler divergence

The Kullback-Leibler divergence between the variational density $q$ and the posterior distribution $p(. \vert x)$ writes:

$$
\begin{aligned}
KL(q\vert \vert p(.\vert x)) &=
\int\limits \log(\frac{q(z)}{p(z \vert x)}) q(z) dz \\ &=
\int\limits \log(\frac{p(x)q(z)}{p(x,z)}) q(z) dz \\ &=
p(x) - \int\limits \log(\frac{p(x,z)}{q(z)}) q(z) dz \\ &=
log(p(x)) - F(x,q)
\end{aligned}
$$

Thus, thus ELBO can be rewritten as $$F(x,q) = log(p(x)) - KL(q\vert \vert p(.\vert x)).$$

Maximizing $F$ with respect to $q$ is the same as minimizing the divergence between the $q$ and the posterior distribution.

## Approximating the posterior distribution

$\newcommand{\PZ}{\mathcal{P(\mathcal{Z})}}$ Let $\PZ$ denote the set of all possible densities defined on the latent space $\mathcal{Z}$. The previous formula gives us a variational definition of the posterior:

$$
p(. \vert x) = \underset{q \in \PZ}{argmax} \space F(x, q)
$$

In variational inference we approximate this true posterior by instead optimizing on a subset of $\PZ$, denoted $\newcommand{\Q}{\mathcal{Q}}$$\Q$.

$$
p(. \vert x) \approx \underset{q \in \Q}{argmax} \space F(x, q)
$$

For instance $Q$ is often taken as a set of gaussian distributions on $Z$.

Using different tricks (e.g. the Mean-Field approximation) allows this familiy of method to scale better than Monte-Carlo estimation, but in contrast doesn't yield an estimate of the exact posterior.

# Conclusion

In this article we have seen how to use variational inference to approximate the posterior distribution in models having unobserved variables.

Note that the hidden variable $z$ can be root nodes in the graphical model, for instance in the case where $z$ are the parameters of the models, or interior nodes, as is the case for instance in Variational Autoencoders. In the latter case the ELBO is used as a computational vehicle to backpropagate to the parameters of a neural network, using the reparameterization trick.

{% bibliography --cited %}
