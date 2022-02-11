---
layout: post

author:
  - Raphaël Romero
bibliography:
  - bibtex.bib
date: February 2022
title: "A geometric view of the Conditional Network Embedding method"
---

Conditional Network Embedding (CNE) is a node embedding method for graphs that has been successfully applied to visualization and prediction.
It allows the user to generate embeddings that respect the network structure, while factoring out prior knowledge known in advance.
Applications of this include visualizing the nodes in a network whithout representing undesired effect, such as for instance having the high degree nodes concentrated in the center of the embedding space.
Such embeddings can also be used to predict links while controlling the influence of sensitive node attributes on the predictions.
This has great interest in producing fair link prediction on social networks for instance.

In what follows we aim to give a comprehensive view of the underlying mechanism that make CNE good at producing embeddings that factor out prior information.

<!--

In what follows we express the Conditional Network Embeddings model as a
statistical model for which the parameter space is the cartesian product
of the space of embedding matrices and regression parameters w.r.t. edge
features $$f_{ij}$$. -->

# Conditional network embedding

# Binary graphs

Conditional Network Embeddings ([@Kang2018]) is a model that maps nodes
from a graph to a Euclidian space conditionally to prior information
encoded in a prior distribution that is computed as the Maximum entropy
distribution under certain constraint statistics.

The final form of the posterior likelihood takes the form
$$P(G|X) = \prod\limits_{i \neq j} \frac{\mathcal{N}_{+}(d_{ij}|\sigma(a_{ij}) P_{ij}(a_{ij})}{\sum\limits_{a\in \{0,1\}} \mathcal{N}_{+}(d_{ij}|\sigma(a_{ij}) P_{ij}(a_{ij})}$$
Where $$\sigma(0)=\sigma_0>0$$ and $$\sigma(1) = \sigma_1>0$$ is the spread
function, with fixed spread hyper-parameters $$\sigma_0$$ and $$\sigma_1$$,
$$P_{ij}$$ is the prior distribution and $$\mathcal{N}_+$$ is the half
normal distribution.

## Prior distribution

In the binary case, the prior distribution is
$$P(G) = \prod_{i\neq j} P_{ij}(a_{ij})$$

where
$$P_{ij} \sim Bernoulli(\frac{exp(\lambda^T f_{ij})}{1 + exp(\lambda^T f_{ij})})$$
$$f_{ij}$$ being the constraint vector associated with edge $$a_{ij}$$.

::: example*
**Example 1**. *For undirected graphs and degree constraints:
$$P_{ij} \sim Bernoulli(\frac{exp(\lambda_i + \lambda_j)}{1+exp(\lambda_i + \lambda_j)})$$\*

_For directed graphs and in and out degree constraints:
$$P_{ij} \sim Bernoulli(\frac{exp(\lambda*i^r + \lambda_j^c)}{1 +exp(\lambda_i^r + \lambda_j^c)})$$*
:::

## Posterior distribution

As seen above, the CNE likelihood depends on the parameters
$$X \in \mathbb{R}^{n \times d}$$ through the matrix of pairwise distances
$$d_{ij}$$ and $$\lambda \in \mathbb{R}^K$$.

The likelihood can be rewritten as

$$P(G;X,\lambda) = \prod\limits_{i \neq j} Q_{ij}(X,\lambda)^{a_{ij}}(1-Q_{ij}(X,\lambda))^{(1-a_{ij})}$$

where $$\begin{aligned}
Q_{ij}(X, \lambda) =& \frac{\sqrt{\frac{2}{\pi \sigma_1^2}} exp(- \frac{d_{ij}^2}{2 \sigma_1^2} + \lambda^Tf_{ij})}{\sqrt{\frac{2}{\pi \sigma_1^2}} exp(- \frac{d_{ij}^2}{2 \sigma_1^2} + \lambda^Tf_{ij}) + \sqrt{\frac{2}{\pi \sigma_2^2}} exp(- \frac{d_{ij}^2}{2 \sigma_2^2})} \\
=& \frac{1}{1 + \frac{\sigma_1}{\sigma_2} exp(- \frac{d_{ij}^2}{2}(\frac{1}{\sigma_2^2} - \frac{1}{\sigma_1^2}) - \lambda^Tf_{ij})} \\
=& sigm(\frac{d_{ij}^2}{2}(\frac{1}{\sigma_2^2} - \frac{1}{\sigma_1^2}) + \lambda^Tf_{ij} + log(\frac{\sigma_2}{\sigma_1})) \\
=& sigm(-K(X_i, X_j) + \lambda^Tf_{ij}) \\\end{aligned}$$

where $$sigm: x \mapsto \frac{1}{1 + e^{-x}}$$ is the sigmoid function and
$$K(X_i, X_j) =\frac{d_{ij}^2}{2}(\frac{1}{\sigma_1^2} - \frac{1}{\sigma_2^2}) - log(\frac{\sigma_2}{\sigma_1})$$

is a distance kernel. It is non-decreasing in the distance between the
embeddings.

This leads us to an alternative expression for the CNE model:

$$a_{ij} \sim Bernoulli(Q_{ij})$$

with
$$logit(Q_{ij}) = \theta_{ij} = \frac{d_{ij}^2}{2}(\frac{1}{\sigma_2^2} - \frac{1}{\sigma_1^2}) + \lambda^Tf_{ij} + log(\frac{\sigma_2}{\sigma_1})$$

In this expression, one could replace the term
$$d_{ij}^2 = ||X_i - X_j||^2$$ by any kernel $$l(X_i,X_j)$$, or by the
similarity yielded by deep graph embedding models such as GraphSage.

## Objective function and gradient

In CNE we first optimize find the optimal $$\lambda$$ parameter value
using by fitting a maximum entropy model (i.e. logistic regression) of
the network given the edge attributes.

Let's write the likelihood of the CNE under parameters $$X, \lambda$$ with
edge features $$f$$:

$$
\begin{aligned}
    p(A|F;X,\lambda) = \prod\limits_{i<j} \frac{exp(a_{ij} \theta_{ij})}{1 + exp(\theta_{ij})}\end{aligned}
$$

So the negative log-likelihood writes:

$$
\begin{aligned}
    L(A|F;X,\lambda) = \sum\limits_{i<j} log(1 + exp(\theta_{ij})) - a_{ij} \theta_{ij}\end{aligned}
$$

While the parameter $$\lambda$$ stays fixed to the Maxent optimum, we use
the chain rule to compute the gradient of this negative log-likelihood
with respect to the embeddings $$X_i$$ for $$i = 1,...,n$$:

Let's denote $$\delta = (\frac{1}{\sigma_1^2} - \frac{1}{\sigma_2^2})$$.
In the original CNE paper we suppose that $$\sigma_2 > \sigma_1$$ so
$$\delta > 0$$.

We have $$\nabla_{X_{i}} (||X_i - X_j||^2) = 2(X_i - X_j)$$ for any
$$i \neq j$$. As a consequence :

$$
\begin{aligned}
\nabla_{X_{i}} (\theta_{ij})
=& \nabla_{X_{i}} (-\delta . \frac{d_{ij}^2}{2} + \lambda^Tf_{ij} + log(\frac{\sigma_2}{\sigma_1})) \\
=& -\delta (X_i - X_j)\\\end{aligned}
$$

We know that the derivative of $$t \mapsto log(1 + t)$$ on $$\mathbb{R}$$ is
the sigmoid function $$\sigma : t \mapsto: \frac{1}{1 + e^{-t}}$$.

So finally:

$$
\nabla_{X_i}(L(A|F; X, \lambda)) = \sum\limits_{j \neq i} \delta.(X_i - X_j) (a_{ij} - \sigma(\theta_{ij}))
\label{gradient}
$$

This expression suggest that CNE can be interpreted as a force-directed
network embedding method. Indeed:

- If $$a_{ij} = 0$$, since $$\delta > 0$$ and
  $$\sigma(\theta_{ij})\in]0,1[$$ $$X_i$$ and $$X_j$$ will be tied by the
  force
  $$\vec{f_{ij}^{(0)}} = \delta \sigma(\theta_{ij})||X_i-X_j|| \vec{u_{i\rightarrow j}}$$,
  where $$\vec{u_{i\rightarrow j}}$$ is a unitary vector oriented from
  $$X_i$$ to $$X_j$$. This correspond to the situation on figure
  [\[fig:repforces\]](#fig:repforces){reference-type="ref"
  reference="fig:repforces"}, where the node $$X_i$$ tries to move away
  from $$X_j$$ in order to minimize its energy.

- If $$a_{ij} = 1$$, $$X_i$$ and $$X_j$$ will be tied by the force
  $$\vec{f_{ij}^{(1)}} = - \delta \sigma(-\theta_{ij})||X_i-X_j|| \vec{u_{i\rightarrow j}}$$,
  where $$\vec{u_{i\rightarrow j}}$$ is a unitary vector oriented from
  $$X_i$$ to $$X_j$$. This correspond to the situation on figure
  [\[fig:attrforces\]](#fig:attrforces){reference-type="ref"
  reference="fig:attrforces"}. Here node $$X_i$$ tries to get closer of
  $$X_j$$ instead, hence the direction of the force.

As a consequence, at each step each node embedding will move according
along a direction which is the sum of all forces that they are submitted
to.

## Gradient subsampling

In expression [\[gradient\]](#gradient){reference-type="ref"
reference="gradient"}, it can be seen that each step can be
computationally intensive since a sum of $$\sim n$$ terms must be
computed.

But one can reduce this complexity by approximating for each node the
resultant of forces by the one coming from a sub sample of the nodes
(i.e. we do neighborhood sampling).

To do this we fix a negative/positive rate (for instance $$r=0.1$$) and we
sample a neighborhood $$S(i)$$ of $$i$$ in which this ratio is observed.

The approximate gradient is then given by :

$$
\nabla_{X_i}(L(A|F; X, \lambda)) = \sum\limits_{j \neq i, j\in S(i)} \delta.(X_i - X_j) (a_{ij} - \sigma(\theta_{ij}))
\label{approxgradient}
$$

## Extension to the exponential family

The derivation of the objective function and gradient can be made more
general, especially for the cases where we do not want to predict edge
presence absence, but more general edge attributes
$$a_{ij} \in \mathcal{A}$$ ($$A = \mathbb{N}$$ for counts,
$$A = \mathbb{R_+}$$ for real-valued weights for instance). To do this we
make use of the Generalized Linear Model, which is based on several
assumptions

- We assume that the edge attributes are generated from a distribution
  belonging the exponential family:
  $$p(a_{ij}|\eta_{ij}) = h(a_{ij})exp(\eta_{ij} a_{ij} - A(\eta_{ij}))$$
  Here A is the cumulant function, $$\eta_{ij}$$ is the natural
  parameter and $$h$$ is the base measure of the distribution.

- The natural parameter of the model is
  $$\eta_{ij} = - \delta.||X_i - X_j||^2 + \lambda^T f_{ij} + \gamma$$
  where $$\gamma$$ and $$\delta$$ are strictly positive real numbers.

Under these assumptions, the negative log-likelihood of the model can be
written as:

$$
\begin{aligned}
    L(a|f, X, \lambda) =& \sum\limits_{i<j} A(\eta_{ij}) - a_{ij} \eta_{ij} - log(h(a_{ij}))\\\end{aligned}
$$

Suppose that we fix the value of $$\lambda$$ and try to minimize this cost
function with respect to the embeddings $$X_i$$.

First we compute the gradient of the natural parameter with respect to
these embeddings:

$$\nabla_{X_i} (\eta_{ij}) = -\delta . (X_i - X_j)$$

Then using the properties of the cumulant we know that the gradient of
$$A$$ is equal to the expectation of the distribution. Using the chain
rule we get :

$$
\begin{aligned}
    \nabla_{X_i} (A(\eta_{ij}))
    =& (\nabla_{X_i}(\eta_{ij})) \nabla A(\eta_{ij})\\
    =& -\delta. (X_i - X_j) . \mu_{ij}\\\end{aligned}
$$

Where $$\mu_{ij} = \mathbb{E}_{a_{ij}' \sim p(.|\eta_{ij})}[a_{ij}']$$ is
the expectation of the distribution. Finally the gradient of $$L$$ can be
expressed as:

$$\nabla_{X_i} L(A|F,X,\lambda) = \sum_{i \neq j} \delta.(X_i - X_j).(a_{ij} - \mu_{ij})$$

So as before, the gradient for each embedding is a resultant of $$n-1$$
forces, one for each other node. For each $$j$$ the component of the force
along the oriented line passing from $$X_i$$ to $$X_j$$ is given by
$$\delta.||X_i - X_j||.(a_{ij} - \mu_{ij})$$. If $$a_{ij}$$ is greater than
the mean, then the embedding $$X_i$$ will be attracted by the embedding
$$X_j$$. Conversely if the $$a_{ij}$$ is lower, $$X_i$$ will tend to move away
from $$X_j$$.

## Expression using the KL divergence between GLM instances

Here we attempt to express the CNE negative log-likelihood in terms of
Kullback-Leibler divergences between several instances of the same
generalized linear model.

Let's introduce some notations to do this.

Let's consider a binary observed variable $$a \in \{0, 1\}$$, and suppose
we would like to derive a Bernoulli model for it.

We'll denote $$\mathbb{P}_{\theta}(a) = exp(\theta a - A(\theta))$$ for
any natural parameter value $$\theta$$. Here
$$A(\theta) = log(1 + exp(\theta))$$ is the cumulant function.

In CNE the natural parameter can be expressed as the sum of two terms,
say $$\beta_{ij}$$ for the pairwise similarity between embeddings and
$$\gamma_{ij}$$ for edge features.

Similarly let's consider that we would like to model our observation
using a natural parameter of the same form:

$$\eta = \beta + \gamma$$

Then the maximum likelihood estimation of $$\mathbb{P}_{\eta}$$ is the
same as minimizing the Kullback-Leibler divergence (i.e. the negative
log-likelihood) from the empirical distribution $$\hat{\mathbb{P}}$$ to
$$\mathbb{P}_{\eta}$$. The latter can be written as:

$$KL(\hat{\mathbb{P}} || \mathbb{P}_{\eta}) = KL(\hat{\mathbb{P}} || \mathbb{P}_{\beta + \gamma}) =  A(\beta + \gamma) - a (\beta + \gamma)$$

Similarly we can write the KL divergences between models
$$\mathbb{P}_{\beta}$$ and $$\mathbb{P}_{\gamma}$$ and the empirical
distribution as:

$$
\begin{aligned}
KL(\hat{\mathbb{P}} || \mathbb{P}_{\beta}) =&  A(\beta) - a \beta \\
KL(\hat{\mathbb{P}} || \mathbb{P}_{\gamma}) =&  A(\gamma) - a \gamma    \end{aligned}
$$

Let's now consider the divergences between the model distributions. We
have:

$$
\begin{aligned}
KL(\mathbb{P}_{\gamma} || \mathbb{P}_{\beta + \gamma}) =& A(\beta + \gamma) - (A(\gamma) + \beta \sigma(\gamma))\\
KL(\mathbb{P}_{\beta} || \mathbb{P}_{\gamma}) =& A(\gamma) - (A(\beta) + (\gamma - \beta) \sigma(\beta))\\
KL(\mathbb{P}_{\gamma} || \mathbb{P}_{\beta}) =& A(\beta) - (A(\gamma) + (\beta - \gamma) \sigma(\gamma))\\\end{aligned}
$$

where $$\sigma$$ is the sigmoid function.

Furthermore, the entropy of the distribution $$\mathbb{P}_{\gamma}$$ can
be written as:

$$H(\mathbb{P}_{\gamma}) = A(\gamma) - \gamma \sigma(\gamma)$$

So combining all of these equations we get:

$$
\label{kldev}
KL(\hat{\mathbb{P}} || \mathbb{P}_{\beta + \gamma}) =
KL(\mathbb{P}_{\gamma} || \mathbb{P}_{\beta + \gamma})
+ KL(\hat{\mathbb{P}} || \mathbb{P}_{\gamma})
+ KL(\hat{\mathbb{P}} ||\mathbb{P}_{\beta})
- KL(\mathbb{P}_{\gamma} || \mathbb{P}_{\beta})
- H(\mathbb{P}_{\gamma})
$$

Note that this equation is symmetric in $$\beta$$ and $$\gamma$$, and that
we have:

$$
\begin{aligned}
KL(\mathbb{P}_{\gamma} || \mathbb{P}_{\beta + \gamma})
- KL(\mathbb{P}_{\gamma} || \mathbb{P}_{\beta})
- H(\mathbb{P}_{\gamma}) =&
KL(\mathbb{P}_{\beta} || \mathbb{P}_{\gamma + \beta})
- KL(\mathbb{P}_{\beta} || \mathbb{P}_{\gamma})
- H(\mathbb{P}_{\beta}) \\
=& -I(\beta, \gamma)\end{aligned}
$$

where we define
$$I(\beta, \gamma) = A(\beta) + A(\gamma) - A(\beta + \gamma) > 0$$ This
term quantifies the interactions between the two effects that are taken
into account in the model.

More precisely if we define the random variables $$B$$ and $$C$$ such that

- $$B \sim \mathbb{P}_{\beta}$$

- $$C \sim \mathbb{P}_{\gamma}$$

- $$B \cap C \sim \mathbb{P}_{\beta + \gamma}$$

Then $$I(\beta, \gamma)$$ is the mutual information between $$B$$ and $$C$$.

On Figure [1](#fig:corrterm){reference-type="ref"
reference="fig:corrterm"} we plot the opposite of this interaction term
as a 3d surface. We see that it is strictly negative for any value of
parameters, and that the maximal values are obtained on the line
$$x = y$$.

![Surface
$$z = - I(x, y) = A(x + y) - (A(x) + A(y))$$](corrterm.png){#fig:corrterm}

::: proof
_Proof._ Equation [\[kldev\]](#kldev){reference-type="ref"
reference="kldev"} can be proved by replacing the KL divergences by
their development above. For the positivity of $$I$$, we remind the
definition of the cumulant function in the exponential family where
$$p(x|\theta) = exp(a \theta - A(\theta))$$ with respect to a base measure
$$h$$. For any $$\theta$$ in the natural parameter space $$\mathcal{N}$$ (the
space where this integral converges).

$$A(\theta) = log(\int exp(a \theta) dh(a))$$

Let's denote $$Z(\theta) = \int exp(a \theta) dh(a)$$ and compute for any
$$\beta, \gamma \in \mathcal{N}$$ such that
$$\beta + \gamma \in \mathcal{N}$$: $$\begin{aligned}
Z(\beta + \gamma) - Z(\beta)Z(\gamma)
=& \int exp(a (\beta + \gamma)) dh(a) - \int exp(s (\beta))dh(s)\int exp(t (\gamma))dh(t) \\
=& \int exp(a (\beta + \gamma)) dh(a) - \int exp(t \beta + s \gamma)dh(s)dh(t) \\
=& \int \mathbbm{1}_{\{s=t\}}exp(s \beta + t\gamma)) dh(s)dh(t) - \int exp(t \beta + s \gamma)dh(s)dh(t) \\
=& \int (\mathbbm{1}_{\{s=t\}} - 1) exp(s \beta + t\gamma)) dh(s)dh(t) < 0\\\end{aligned}$$
The last inequality comes from the positivity of the integral and the
fact that the integrand is strictly negative.

So this gives us $$\frac{Z(\beta) Z(\gamma)}{Z(\beta + \gamma)} > 1$$

Taking the log of this expression gives us that $$I(\beta, \gamma) > 0$$ ◻
:::

Equation [\[kldev\]](#kldev){reference-type="ref" reference="kldev"} can
be written in the following compact form :

$$
\begin{aligned}
    KL(\hat{\mathbb{P}} || \mathbb{P}_{\beta + \gamma}) =  KL(\hat{\mathbb{P}} || \mathbb{P}_{\gamma})
+ KL(\hat{\mathbb{P}} || \mathbb{P}_{\beta})
- I(\beta, \gamma)\end{aligned}
$$

Going back to the CNE formalism, here the term
$$KL(\hat{\mathbb{P}} || \mathbb{P}_{\gamma})$$ would be the Maxent
objective function, $$KL(\hat{\mathbb{P}} || \mathbb{P}_{\beta})$$ would
be the likelihood of the embedding model fitted without any Covariate
term, and $$I(\beta, \gamma)$$ encodes the interactions between the
covariates and the embeddings.

Because of the $$-$$ sign in front of $$I$$, minimizing this expression with
respect to both $$\gamma$$ and $$\beta$$ would automatically tend to
maximize the interaction term.

Conversely CNE process consists in training
$$KL(\hat{\mathbb{P}} || \mathbb{P}_{\gamma})$$ separately without the
interaction term, and then fix the $$\gamma$$ value to find the $$\beta$$
for which this interaction is reduced compared to if we would perform a
full joint training.

Indeed the objective of the CNE with respect to $$\beta$$ is

$$KL(\hat{\mathbb{P}} || \mathbb{P}_{\beta})  - I(\beta, \gamma)$$

It can be seen as a _conditional_ objective: removing this mutual
information term is similar to conditioning in probability terminology.

Let's make this conditional objective more flexible by adding a weight
$$\alpha$$ in front of the mutual information term.

$$KL(\hat{\mathbb{P}} || \mathbb{P}_{\beta})  - \alpha I(\beta, \gamma)$$

Then the hyperparameter $$\alpha$$ should control to what extend we want
to remove the information already learned by $$\gamma$$ from the
embeddings $$\beta$$.

# Expression as a Bayesian Network

The various conditional dependencies that are present in the CNE can be
summerized using the means of a bayesian network. In the binary
undirected case, this network would contain:

- $$\frac{n (n-1)}{2}$$ nodes to model the values $$a_{i,j}$$ in the
  adjacency matrix

- $$\frac{n (n-1)}{2}$$ features vector to model the values $$f_{i,j}$$ of
  the edge features

- $$n$$ nodes containing the embeddings $$X_{i}$$.

- $$1$$ node containing the vector $$\lambda$$

Although the full network would be more complex to represent, we can
easily represent the neighborhood of a given each $$a_{ij}$$ nodes:

On Figure [\[fig:cnegm\]](#fig:cnegm){reference-type="ref"
reference="fig:cnegm"} the $$X_i$$ and $$X_j$$ are also connected to all the
others $$a_{ij}$$ and $$f_{ij}$$ in the model, but since they are root nodes
(they don't have parents nodes), the local representation is sufficient
to derive the joint probability.

Let's suppose we postulate prior distributions $$p(X)$$ and $$p(\lambda)$$
on the embeddings and on the parameters $$\lambda$$. For instance:
$$X_i \overset{iid}{\sim} \mathcal{N}(0, \Sigma)$$ and
$$\lambda \sim \mathcal{N}(0, \Gamma)$$ with
$$\Sigma, \Gamma \in S_n^+(\mathbb{R})\times S_K^+(\mathbb{R})$$
respectively.

The joint probability factorizes with respect to the graphical model
above as :
$$p(A,F, X, \lambda) = \prod\limits_{i<j} p(a_{ij}|f_{ij}, x_{i}, x_j, \lambda) p(f_{ij})p(\lambda)p(x_i)p(x_j)p(\lambda)$$

Formally, the posterior distribution on the parameters would then write
as:

$$p(X,\lambda|A,F)=\frac{p(A,F|X, \lambda) \times  p(X)p(\lambda) }{\int\limits_{X,\lambda} p(A,F|X, \lambda) \times  p(X)p(\lambda)} dX d\lambda$$
