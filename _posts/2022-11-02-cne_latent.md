---
title: Conditional Network Embedding, a Latent Space Distance perspective
header-includes:
  - \usepackage{bbm}
  - \usepackage{amsmath,amssymb}
layout: post
author:
  - Raphaël Romero
bibliography:
  - bibtex.bib
date: February 2022
---

<!--more-->

# Introduction

Conditional Network Embedding (CNE) {% cite KangLB19 %} is a node embedding method for graphs that has been successfully applied to visualization and prediction. It allows the user to generate node embeddings that respect the network structure, while factoring out prior knowledge known in advance. Applications of this include visualizing the nodes in a network without representing undesired effect, such as for instance having the high degree nodes concentrated in the center of the embedding space. The resulting embeddings can also be used to predict links while controlling the influence of sensitive node attributes on the predictions. This has great interest in producing fair link prediction on social networks, such as in {% cite buyl20a %}.

In what follows we aim to give a comprehensive view of the underlying mechanism that make CNE good at producing embeddings that factor out prior information.

<!--

In what follows we express the Conditional Network Embeddings model as a
statistical model for which the parameter space is the cartesian product
of the space of embedding matrices and regression parameters w.r.t. edge
features $$f_{ij}$$. -->

# Conditional network embedding

Conditional Network Embedding is a graph embedding method.

Given an undirected graph $G=(U,E)$ where $U$ is the set of nodes and $E\subset U\times U$ is the set of nodes it yields a mapping from the set of nodes to a $d$-dimensional space:

$$
\begin{aligned}
CNE \colon U &\rightarrow & \mathbb{R}^d \\ u &\mapsto & z_u \\
\end{aligned}
$$

# Factoring out prior information in embeddings

$\newcommand{\norm}[1]{\vert \vert #1 \vert \vert }$ In CNE, we suppose that we have encoded our prior expectations about an observed graph $\hat{G}$ into a MaxEnt distribution(see [my post about Maxent]({{site.url}}/{% post_url 2022-11-02-maxent %}) or the paper {% cite debie2010maximum %}). Moreover, we suppose that each node $i \in U$ is represented by an (unknown) embedding vector $z_i \in \mathbb{R}^d$, and that for two nodes $i \neq j$, their connection only depends on the embedding through the euclidean distance between their embeddings $d\_{ij} = \norm{z_i-z_j}$.

Based on that, CNE uses Bayes' rule to define the link probability conditioned on the MaxEnt distribution:

$$
P_{ij}(a_{ij}|z_i, z_j)= \frac{
\mathcal{N}_{+}(d_{ij} | s(a_{ij}))
P_{ij}(a_{i,j})
}{
\sum\limits_{a \in \{0,1\}}
\mathcal{N}_{+}(d_{ij} | s(a))
P_{ij}(a)
}
$$

where

- $d_{ij} = \vert\vert z_i-z_j \vert\vert$ is the euclidean distance between embeddings $z_i$ and $z_j$.
- $\mathcal{N}_{+}(d\vert s(a))$ denotes a half normal density with spread parameter s(a).
- $s$ is a spread function such that $s_0=s(0)>s(1)=s_1$
- $P_{ij}(a)$ is the MaxEnt prior Bernoulli distribution

Thus, CNE postulates a distribution over the distance between embeddings, such that the distances between embeddings of non-edges are more spread around 0 than for edges.

Finally, the probability of full graph $G$ is defined as the product of the independent link probabilities:

$$
P(G\vert Z) =\prod_{i\neq j}P_{ij}(a_{ij}|z_i, z_j)
$$

### Retrieving the link Bernoulli probabilities

As seen before, the full likelihood of a graph under the CNE model can be written as product of independent probabilities, one for each node pair. As the link indicator $a_{ij}$ between each node pair $ij$ is a Bernoulli random variable, one can transform the expression in order to retrieve the Bernoulli probabilities.

Indeed, it can be shown that the edge link probabilties can be rewritten as: $$P_{ij}(a_{ij} \vert z_i, z_j) =  Q_{ij}^{a_{ij}}(1-Q_{ij})^{(1-a_{ij})}$$

Where:

- $Q_{ij} = \sigma \left(\alpha + \lambda^Tf_{ij} - \beta.\frac{d\_{ij}^2}{2} \right)$
- $\alpha=\log(\frac{s_1}{s_0})$ is a non-negative constant.
- $\beta=(\frac{1}{s_1^2} - \frac{1}{s_0^2}) \geq 0$ is a scaling constant.
- $\sigma$ still denotes the sigmoid function

#### Proof

In order to retrieve this form, we first recall the form of the Half-Normal distribution:

$$
\begin{aligned}
p_{\mathcal{N}_{+}(.\vert s)}(d) = \sqrt{\frac{2}{\pi s^2}} exp(- \frac{d^2}{2 d^2})
\end{aligned}
$$

Moreover, the MaxEnt prior distribution writes:

$$
\begin{aligned}
P_{ij}(a_{ij})=\frac{exp(\lambda^Tf_{ij}(G))}{1+exp(\lambda^Tf_{ij}(G))}
\end{aligned}
$$

Since $P_{ij}(a_{ij} \vert z_i, z_j)$ is a Bernoulli probability, we have $Q_{ij} = P_{ij}(1 \vert z_i, z_j)$

Injecting $a_{ij}=1$ in the expression of $P_{ij}(a_{ij} \vert z_i, z_j)$ and simplifying gives:

$$
\begin{aligned}Q_{ij}=& \frac{
\sqrt{\frac{2}{\pi s_1^2}}
exp(- \frac{d_{ij}^2}{2 s_1^2} + \lambda^Tf_{ij})
}{
\sqrt{\frac{2}{\pi s_1^2}}
\exp(- \frac{d_{ij}^2}{2 s_1^2} + \lambda^Tf_{ij}) +
\sqrt{\frac{2}{\pi s_0^2}}
\exp(- \frac{d_{ij}^2}{2 s_0^2})
} \\ = &
\frac{1}{
  1 +
\exp\left(- \frac{d_{ij}^2}{2}(\frac{1}{s_0^2} - \frac{1}{s_1^2}) - \lambda^Tf_{ij} - log(\frac{s_0}{s_1})\right)
} \\ =&
\sigma(\lambda^Tf_{ij} + log(\frac{s_0}{s_1}) - \frac{d_{ij}^2}{2}(\frac{1}{s_1^2} - \frac{1}{s_0^2})) \\
\end{aligned}
$$

where $\sigma:x \mapsto \frac{1}{1+exp(-x)}$ is the sigmoid function.

### Connection with Latent space models for graphs

As we see, the independent link logits logit in CNE are given by subtracting the scaled distance between embeddings to prior terms and a constant bias: $$logit(Q_{ij})=C+ \lambda^Tf_{ij} - D . d_{ij}^2$$ where $C= log(\frac{s_0}{s_1})$ and $D=0.5*(\frac{1}{s_1^2} - \frac{1}{s_0^2})$

(The logit is defined as the inverse of the sigmoid function: $\sigma(logit(p)) = logit(\sigma(p))=p$)

Intuitively, the term $\lambda^Tf_{ij}$ encodes a prior similarity value between $i$ and $j$ that doesn't need to be represented by a small distance between the embeddings $z_i$ and $z_j$.

This type of statistical model has been studied in a variety of previous work, in the name of Latent Space Distance Models {% cite Hoff2002 Turnbull2019 Ma2020a %}.

The common principle of this type of method is use the latent distance between vector representations as sufficient statistics for the link indicator variable.

# Example with the degree and edge features as prior.

Here we given an example of CNE model where we retrieve the Bernoulli probabilities $Q_{ij}$ given some prior statistics.

We consider a simple example of CNE, where the MaxEnt statistics used are:

- The degree of each node $i$: $f_i^{(degree)}(G) = \sum\limits_{j\in \cal{N}(i)} a_{ij}$ where $\cal{N}(i)$ is the set of neighbors of $i$. This leads to $n$ statistics at the graph level. For each edge $ij$ the corresponding edge-level statistics vector $f_{ij}$ are given by $[E_i^n \vert\vert E_j^n]$, where for each node $i$, $E_i^n$ is the n-dimensional one-hot encoding of the node $i$ and $\vert\vert$ represents the concatenation operation. Denoting $\alpha \in \mathbb{R}^{2n}$ the vector of coefficients associated to these degree statistics, the corresponding logit value is equal to $$\alpha^Tf_{ij}=\alpha_i + \alpha_j$$

- Some edge-level features $x_{ij}$. We denote $\theta$ the associated coefficient and the logit values coming from it are equal to : $$\theta^T x_{ij}$$

So by stacking all these features, we get the following prior term:

$$\lambda^Tf_{ij}=\alpha_i + \alpha_j + \theta^T x_{ij}$$

The CNE Bernoulli probabilities are thus equal to:

$$Q_{ij} = \sigma \left(C + \alpha_i + \alpha_j + \theta^T x_{ij} - D. d\_{ij}^2 \right) $$

<!--
# Visual explanation

In order to geometrically explain how CNE factors out prior knowledge, a possible approach is to imagine the (random) edges as Bernoulli random variables, to make them deterministic variables conditioned on the embeddings.

### Deterministic version of the random graphs above.

The sigmoid function is a smooth version of a non-continuous function, the Heaviside step function, given by $h(x) = \mathbb{1}_{\{x>0\}}$.
This one yields an activation equal to 1 for positive inputs and 0 for negative inputs.

![Heaviside]({{site.url}}/figures/sigmoid_vs_heaviside.png)
_The heaviside function in red, and the sigmoid function in green_

Let's consider a CNE model, where we use as constraints the degrees of each nodes, as well as other features.
The CNE expression looks like:


$$
Q_{ij} = \sigma \left(2 \gamma +\alpha_i + \alpha_j+ \theta^T x_{ij} - \vert\vert z_i-z_j\vert\vert^2 \right)
$$

In the deterministic CNE expression, the link indicators would then look like:

$$
a_{ij} =h\left(2\gamma +\alpha_i + \alpha_j+ \theta^Tx_{ij} - \vert\vert z_i-z_j\vert\vert \right)
$$

This has a natural visual interpretation, as shown in the following image
![CNE-DEG]({{site.url}}/figures/cne_deg1.png)

As can be seen, each embedding $z_i$ is endowed with a disk $D_i$of radius $\alpha_i+\gamma$ such that the minimum distance between $D_i$ and $D_j$ in order for the nodes to connect is $\theta^T x_{ij}$.

If the prior similarity is high, the the disk need not be too close for the connection to form. As a consequence, the embeddings will not encode the prior information. -->

## References

{% bibliography --cited %}
