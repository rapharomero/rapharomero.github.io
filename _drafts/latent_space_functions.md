---
layout: post
title: Investigating Different distance function for Latent Distance graph models
author:
  - Raphaël Romero

header-includes:
  - \usepackage{mathbbm}
bibliography:
  - "bibtex.bib"
date: October 2020
---

In this post, we consider Latent Space models for graphs, and investigate the impact of the distance function used on the embedding space.

<!--more-->

# Introduction

Latent Space models for graphs are defined such that the independent edge link probabilities are given by functions of the distance between some embeddings. $\newcommand{\norm}[1]{\vert\vert #1 \vert\vert}$

$$
\begin{align}
\label{model}
y_{ij} &\sim Bernoulli(\theta_{ij}) \\ \theta_{ij} &= \sigma(x_{ij}) \\  x_{ij} &= \gamma- g(\norm{x_i - x_j}^2)
\end{align}
$$

where $\sigma$ is the sigmoid function .

The function g can be any non-decreasing, non-negative smooth function. For instance:

- If g is the identity function: $x_{ij} = \gamma- \norm{x_i - x_j}^2$
- If g is the square root function: $x_{ij} = \gamma- \norm{x_i - x_j}$
- If g is the log: $x_{ij} = \gamma- 2\log(\norm{x_i - x_j})$

We would like to investigate the impact on this distance function on the embeddings found by performing Maximum Likelihood Estimation of the model, given an observed graph.

# Likelihood and gradient of the model

The likelihood of a given observed undirected graph $\hat{G}$ with adjacency matrix $A={a_{ij}}$ is given by:

$$p(\hat{G}) = \prod\limits_{i<j} \theta_{ij}^{y_{ij}} (1-\theta_{ij})^{1-y_{ij}}$$

Hence we get the following negative log-likelihood, as a function of the embeddings $z_i$:

$$L(z) = \sum\limits_{i<j} log(1+ exp(x_{ij}))  - y_{ij} x_{ij}$$

# Gradient

For a given node $i$, we compute the gradient of the loss function with respect to the embedding $z_i$.

This one is given by:

$$\nabla\_{z_i}L(z) = \sum_{j\neq i} (\nabla_{z_i}x_{ij}) (a_{ij} - \sigma(x_{ij}))$$

Moreover, using the chain rule gives us the gradient of the logit $x_{ij}$ with respect to the embeddings:

$$\nabla_{z_i}x_{ij} = -2(z_i - z_j) g'(\norm{z_i-z_j}^2)$$

So finally we get the following gradient:

$$\nabla\_{z_i}L(z) = \sum_{j\neq i} -2(z_i - z_j) g'(\norm{z_i-z_j}^2) (a_{ij} - \sigma(x_{ij}))$$

# Interpretation in terms of forces

As we see in the previous expression, the gradients with respect to the embeddings can be view as a set of forces pulling or repulsing the embeddings away from each other depending on whether the corresponding nodes are linked in the graph or not.

- If the nodes $i$ and $j$ are connected, we get $$a_{ij} - \sigma(x_{ij}) = 1 - \sigma(x_{ij}) = \frac{1}{1+\exp(x_{ij})}$$.

{% bibliography --cited %}
