---
title: Survival Analysis applied to Graphs
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

# Project Description

# Initial Idea

In the AIDA group, a lot of previous research has been focused on graph analysis, with an emphasis on link prediction. For this latter task the idea is to use the graph structure in order to predict the presence/absence of links between two nodes in the graph.

Based on the paper [Bender:2021](https://www.notion.so/Bender-2021-d15bde1e3ae949d79a949d5ef9d22fa6) , we figured that survival analysis can be used to use the temporal dynamic in order to improve link prediction on graphs.

The intuition is that any event in a temporal network, such as an instantaneous connection, can be thought as the birth of a virtual individual, that will die and be reborn at the next event.

As such, we can use survival analysis to predict how much time will elapse between two events, given the conditions in which the first event occured.

$$
t_{n+1}\sim p_{\theta}(.|x_{t_n})
$$

The notion of censoring can be used in case the next event never happens.

In the paper [Bender:2021](https://www.notion.so/Bender-2021-d15bde1e3ae949d79a949d5ef9d22fa6), the authors propose to use a piecewise constant hazard as a modelling assumption for time to event data.

Indeed, such assumption allows to cast the estimation problem into a Poisson Regression, which is supported by a wide-range of machine learning tools available out-of-the box.

# Shortcomings and propostion

A shortcoming of the idea proposed above is that the hazard function is not directly available from it for any given node pair for instance. Indeed, we only postulate a distribution for the waiting time to a next event given the previous event and the associated features.

In general, we need to postulate the hazard function of any event $e$ as a function of time.

## Case of dyadic events

We restrict ourselves to dyadic events for now. For dyadic events, the events are labelled by a dyad index $e=(u,v)$. For each of them we need to postulate a hazard function.

To do this, we suppose that conditionned on a previous event, the distribution is the one proposed in the original idea, but that in the absence of any event, the event still has a base rate function $\mu_e$

As a consequence the hazard function for any event $e$ is given by

$$
\lambda_e(t) = \mu_e(t)+ \sum_{t_n^e<t} \mathbb{1}_{t\in]t_{n}^e, t_{n+1}^e]} g_{\theta}
(t-t_n|x_{t_n^e}^e)
$$

In this expression, we can see that the hazard value is the sum of a base hazard $\mu_e(t)$ and a transition hazard $g_{\theta}
(t-t_n|x_{t_n^e}^e)$. The sum of the right-hand side will indeed only have one non-zero term, for the last event before $t$.

<aside>
💡 This formulation is in analogy with the one from Hawkes process, which postulates a contribution of each previous event for the dyad as an exponentially decayed weight:

$$
\begin{align*}
\lambda_e(t)
&= \mu_e(t)+ \sum_{t_n^e<t} g_{\theta}
(t-t_n) \\
&= \mu_e(t)+ \int_0^t g_{\theta}
(t-s)dN^e(s) \\
\end{align*}
$$

Here $g_{\theta}$ is a kernel function. This expression can be thought of as the convolution of the differential of the counting process through the kernel function.

</aside>

## Likelihood function

Given a sequence of timestamped events $(e_1, t_1),...,(e_M,t_M)$ where $t_1<...<t_M$ are the time stamps, and $e_m, m=1,...,M$ are the involved marks, the likelihood function of the full history writes:

$$
\begin{align*}
p(H)
&= \prod_{m=1}^{M}\lambda(e_m,t_m)\prod_{e\in\mathcal{E}} exp(-\int_{t_{m-1}}^{t_m}\lambda(e, s))ds \\
&= \prod_{m=1}^{M}[\lambda(e_m,t_m)exp(-\int_{t_{m-1}}^{t_m}\lambda(e_m, s) ds)\prod_{e\in\mathcal{E}, e\neq e_m} exp(-\int_{t_{m-1}}^{t_m}\lambda(e, s) ds)] \\

\end{align*}
$$

Each positive event contributes with an exponential likelihood term, and for each of them each negative event has a contribution equal to the survival function of the associated hazard.

In practice, it might be infeasible to include a negative term for all the negative edges for each positive term. To tackle this problem , an i

To tackle this problem , an idea can be to sample a certain number of negative neighbors for each positive one.

For instance, if $(1, 2, 0 .1 )$  is a positive edge, we can use the edges $(1,4,0.1),  (1,6,0.1),  (1,10,0.1)$ ... as negative edges to contrast with the positive one.

## Examples

### 1. Constant base rate + constant transition hazard

One example of specification for the base rate and transition rate can be to have $\mu_e(t) = \mu_0$ and $g_{\theta}(t-t_n|x_{t_n}) = exp(\gamma_0 + \beta_u + \beta_v)$ where $e=(u,v)$ is the considered dyadic event and $\beta_u, \beta_v$ are some trainable sociality coefficients.

$$
\lambda_e(t) = exp(\mu_0+ \mathbb{1}_{N_e(t)>=1}(\gamma_0 + \beta_u+\beta_v))
$$

$\lambda_e(t) = exp(\gamma_0 + \alpha_u + \alpha_v)$

### 2. Dyad-specific base rate and constant transition hazard

$$
\lambda_e(t) = exp(\gamma_0 + \alpha_u + \alpha_v)+ \mathbb{1}_{N_e(t)>=1}exp(\gamma_1 + \beta_u+\beta_v)
$$

### 3. Dyad-specific base rate and transition hazard depending on the features at the time of the last event

$$
\lambda_e(t) = exp(\gamma_0 + \alpha_u + \alpha_v)+ \sum_{t_n^e<t} \mathbb{1}_{t\in]t_{n}^e, t_{n+1}^e]}  exp(\theta_0 + \theta^T x_{t_n^e}^e)
$$

<aside>
💡 Here the features at the time of the last event $x_{t_n^e}^e$ can be any set of time varying features and their concatenation. 
For instance, we can use $x_{t_n^e}^e = \textbf{1}_n^u||\textbf{1}_n^v$ where $\textbf{1}_n^u$ is a $n$-dimensional one hot encoding of the index of the node u ($n$ being the number of  nodes), and retrieve the sociality coefficients since:
$\theta^Tx_{t_n^e}^e = \theta_u + \theta_v$.

</aside>

### 4. Dyad-specific base rate and piecewise constant transition hazard

$$
\lambda_e(t) = exp(\gamma_0 + \alpha_u + \alpha_v)+ \sum_{t_n^e<t} \mathbb{1}_{t\in]t_{n}^e, t_{n+1}^e]}  g_{\theta}
(t-t_n|x_{t_n^e}^e)
$$

where

$$
 g_{\theta}
(t-t_n|x_{t_n^e}^e) = \sum_{j=1}^J \mathbb{1}_{I_{j}}(t-t_n) exp(\theta_j^T x_{t_n^e}^e
)
$$

We can see that the previous case corresponds to a piecewise constant transition hazard with only one interval.

Potentially, using several intervals can lead to an increased accuracy. Indeed, we can suspect that the transition hazard will not be constant , but vary after a certain time since the last event .

In this case , the cumulative hazard function writes:

$$
\begin{align*}
\int_{t_{m-1}}^{t_m} \lambda_e(s)ds
&= (t_m - t_{m-1})exp(\gamma_0+\alpha_u+\alpha_v) \\
&+ \int_{t_{m-1}}^{t_m}\sum_{t_n^e\leq t_{m-1}}
\mathbb{1}_{s\in]t_{n}^e, t_{n+1}^e]}  g_{\theta}
(s-t_n^e|x_{t_n^e}^e)ds

\end{align*}
$$

# Evaluation

<aside>
💡  We consider some data of the type  $H=(e_1,t_1), ...,(e_M,t_M)$. 
This type of history is a sample of a temporal marked point process , with sampled times $t_m$ and marks $e_m\in \mathcal{E}$.

The full likelihood of such a history is :

$$
p(H)= \prod_{m=1}^M \lambda_{e_m}(t_m) \prod_{e\in\mathcal{E}}exp(-\int_{t_{m-1}}^{t_m}\lambda_e(s)ds))
$$

Where we define a null event at time $t_0$.

In practice, the negative log-likelihood of $H$ can be approximated by selecting a subset of the edges $\tilde{\mathcal{E}}_m$for each event positive $e_m$.
It then writes

$$

\begin{align*}
L(H) = -\sum_{m=1}^M [
(C_m\sum_{e \in \tilde{\mathcal{E}}_m}\int_{t_{m-1}}^{t_m}\lambda_e(s) ds )- log(\lambda_{e_m}(t_m))]

\end{align*}


$$

Where $C_m = \frac{|\mathcal{E}|}{|\tilde{\mathcal{E}_m}|}$ is a scaling factor.

</aside>

## ROC-AUC metric

One way to evaluate the model is to measure how well the hazard function ranks the positive events compared to the negative ones.

To compute it, we compute for each event ($e_m,t_m$)in the test set their score $\lambda(e_m,t_m)$ as well as the scores of a certain number of $\lambda(\tilde{e}_m, t_m)$ for $\tilde{e}_m \in \tilde{\mathcal{E}}_m$.

This allows us to build a dataset of pairs $(x_i,y_i)$, where $x_i$ is the score of the event and $y_i$ is the indicator that the event is a positive /negative one.

<aside>
💡

The ROC-AUC works as follow:

For each threshold $\tau \in [0,1]$ we define a decision rule $f_{\tau}(x_i) = \mathbb{1}_{x_i\geq\tau}=\tilde{y_{i}}$

We can compare these predicted labels to the true ones, by defining some rates:

- False Positive Rate: t e proportion of items such that $y_{i}=0$ and $\tilde{y_{i}}=1$
- True Positive Rate: the proportion of items such that such that 1 and $\tilde{y_{i}}=1$

The ROC- Curve is a parametric curve defined by the points $(FP_{\tau}, TP_{\tau})$ for each value of $\tau$.

In the ideal case, this curve collapses to the point $(0,1)$ since there should be 0 false positives and only true positives.

In the random case, the decision rule yields as many TP as FP, so the curve is the diagonal between $(0,0)$ and $(0,1)$

</aside>

## Mean Reciprocical Rank

For each score $\lambda(e_m,t_m)$ and the associated negative event scores $\{\lambda(\tilde{e}_m, t_m);\tilde{e}_m \in \tilde{\mathcal{E}}_m\}$, we can compute the rank of the positive score in the list of all scores.

The mean reciprocical rank is the mean of the inverese of the scores across all the events.
