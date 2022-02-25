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

Latent Space models for graphs are defined such that the independent edge link probabilities are given by functions of the distance between some embeddings. $\newcommand{\norm}[1]{\vert\vert #1 \vert\vert}$. Denoting $i,j$ some node indices, $z_i \in \mathbb{R}^d$ some node embeddings of dimension $d>0$.

$$
\begin{align}
\label{model}
y_{ij} &\sim Bernoulli(\theta_{ij}) \\ \theta_{ij} &= \sigma(x_{ij}) \\  x_{ij} &= \gamma- g(\norm{z_i - z_j}^2)
\end{align}
$$

where $x_{ij}$ are the logits of the model, $\theta_{ij}$ are the edge link probabilities and  $\sigma$ is the sigmoid link function. 

The function g can be any non-decreasing, non-negative smooth function. For instance:

- If g is the identity function: $x_{ij} = \gamma- \norm{z_i - z_j}^2$
- If g is the square root function: $x_{ij} = \gamma- \norm{z_i - z_j}$
- If g is the log: $x_{ij} = \gamma- 2\log(\norm{z_i - z_j})$

We would like to investigate the impact on this distance function on the embeddings found by performing Maximum Likelihood Estimation of the model, given an observed graph.

# Likelihood and gradient of the model

The likelihood of a given observed undirected graph $\hat{G}$ with adjacency matrix $A={a_{ij}}$ is given by:

$$p(\hat{G}) = \prod\limits_{i<j} \theta_{ij}^{y_{ij}} (1-\theta_{ij})^{1-y_{ij}}$$

Hence we get the following negative log-likelihood, as a function of the embeddings $z_i$:

$$L(z) = \sum\limits_{i<j} log(1+ exp(x_{ij}))  - y_{ij} x_{ij}$$

# Gradient

For a given node $i$, we compute the gradient of the loss function with respect to the embedding $z_i$.

This one is given by:

$$\nabla\_{z_i}L(z) = \sum_{j\neq i} (\nabla_{z_i}x_{ij}) (y_{ij} - \sigma(x_{ij}))$$

Moreover, using the chain rule gives us the gradient of the logit $x_{ij}$ with respect to the embeddings:

$$\nabla_{z_i}x_{ij} = -2(z_i - z_j) g'(\norm{z_i-z_j}^2)$$

So finally we get the following gradient:

$$\nabla\_{z_i}L(z) = \sum_{j\neq i} 2(z_i - z_j) g'(\norm{z_i-z_j}^2) (y_{ij} - \sigma(x_{ij}))$$

# Interpretation in terms of forces

As we see in the previous expression, the gradients with respect to the embeddings can be view as a set of forces pulling or repulsing the embeddings away from each other depending on whether the corresponding nodes are linked in the graph or not.

- If the nodes $i$ and $j$ are connected (i.e. $y_{ij}=1$), we get :
  $$y_{ij} - \sigma(x_{ij}) = 1 - \sigma(x_{ij}) = \frac{1}{1+\exp(x_{ij})}>0.$$
  So the associate gradient term will be the following *attractive force*
  $$
  \begin{aligned} 
  \vec{f_{ij}^{+}} &= 2(z_i - z_j) g'(\norm{z_i-z_j}^2) (y_{ij} - \sigma(x_{ij})) \\ &= (z_i - z_j) (\frac{2 g'(\norm{z_i-z_j}^2)}{1+\exp(x_{ij})})
  \end{aligned}
  $$
  Indeed, since $g$ is non-decreasing we have $\frac{2 g'(\norm{z_i-z_j}^2)}{1+\exp(x_{ij})} >=0$, so this vector is oriented from the embedding $z_j$ to the embedding $z_i$, hence the term "attractive".
  Later, we might be interested in how the intensity of this force scales with the distance between embeddings.
- If the nodes $i$ and $j$ are not connected, we have $y_{ij} - \sigma(x_{ij}) = -\sigma(x_{ij})$ the embeddings are connected by the following $repulsive force$ (essentially pushing away $z_j$ from $z_i$): 
  $$
  \begin{aligned} 
  \vec{f_{ij}^{-}} = - (z_i - z_j) (\frac{2 g'(\norm{z_i-z_j}^2)}{1+\exp(-x_{ij})})
  \end{aligned}
  $$


Denoting the *sign* variable $s_{ij} = 1$ if $y_{ij}=1$ and $s_{ij} = -1$ if $y_{ij}=0$, we get the following compact formula for this force term:
$$
  \begin{aligned} 
  \vec{f_{ij}} = s_{ij} (z_i - z_j) (\frac{2 g'(\norm{z_i-z_j}^2)}{1+\exp(s_{ij} x_{ij})})
  \end{aligned}
  $$


# Examples 

Using different distance functions, we can derive the attractive and repulsive forces to have an idea of their intensity.

### Identity distance function
In the case where $g$ is simply the identity function, we get a signed force equal to 
$$\vec{f_{ij}} =  s_{ij}(z_i-z_j)\frac{2}{1+\exp(s_{ij} (\gamma - \norm{z_i-z_j}^2))}$$
Thus, in that case the norm of the force is given by 

$$\norm{\vec{f_{ij}}} = \frac{2\norm{z_i-z_j}}{1+\exp(s_{ij} (\gamma - \norm{z_i-z_j}^2))}$$ 

- For positive edges ($s_{ij}=1$), this force reaches its minimum when the embeddings match, and will tend to infinity exponentially in the squared distance between embeddings:
$$\norm{\vec{f_{ij}}} \sim 2\norm{z_i-z_j}exp(\norm{z_i-z_j}^2)$$ when $\norm{z_i-z_j} \rightarrow +\infty$.
- For negative edges ($s_{ij}=-1$), this force becomes decreasing in the distance, and tends to $0$ when $\norm{z_i-z_j} \rightarrow +\infty$.


### Square root distance functions
If $g$ is the squared root function, we get a signed force equal to 
$$\vec{f_{ij}} =  \frac{s_{ij}(z_i-z_j)}{\norm{z_i - z_j}(1+\exp(s_{ij} (\gamma - \norm{z_i-z_j}^2)))}$$

The norm of this force term writes:
$$\norm{\vec{f_{ij}}} = \frac{2}{1+\exp(s_{ij} (\gamma - \norm{z_i-z_j}^2))}$$  
This has the following assymptotic behavior:
- when $\norm{z_i-z_j} \rightarrow +\infty$.


### Log distance functions
If $g$ is the log, we get 
$$\vec{f_{ij}} =  \frac{2s_{ij}(z_i-z_j)}{\norm{z_i - z_j}^2(1+\exp(s_{ij} (\gamma - \norm{z_i-z_j}
^2)))}$$


In that case, the norm of the force is 

$$\norm{\vec{f_{ij}}} = \frac{2}{\norm{z_i - z_j}(1+\exp(s_{ij} (\gamma - \norm{z_i-z_j}
^2)))}$$


For both positive and negative edges, this one tends to $+\infty$ when the distance tends to $0$


We see that the first order derivative of the distance function has an impact on the type of force 


{% bibliography --cited %}
