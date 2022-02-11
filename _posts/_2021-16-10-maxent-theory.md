---
layout: post

author:
  - Raphaël Romero

header-includes:
  - \usepackage{mathbbm}
bibliography:
  - "bibtex.bib"
date: October 2020
title: Maximum entropy models for graphs
---

# General Case

In this section, as in we derive the form of maximum entropy
distribution in the general case. We use the derivations made in
[@debie2010maximum] in a large extent.

Let $$(\mathcal{G}, \mathcal{T})$$ be a measurable space containing a
certain type of graphs on a given set of nodes $$1,...,n$$, and
$$\mathcal{P(G)}$$ the set of all probability distributions over this
space.

The Maxent convex optimization problem consists of finding the elements
of $$\mathcal{P}(\mathcal{G})$$ that have maximal entropy, while
respecting a certain number of constraints, expressed through various
_statistics_ or _observables_. Its general expression is the following:

$$
\left\{
\begin{array}{cc}
        \max\limits_{P} -\sum\limits_{G \in \mathcal{G}} P(G) log(P(G)) \\
        s.t. \left\{
        \begin{array}{cc}
              \forall k = 1,...,K,  \mathbb{E}_P [f_k(G)] = c_k \\
              \sum\limits_{G \in \mathcal{G}} P(G) = 1
        \end{array}

        \right.
\end{array}
\right.
$$

In this expression, the $$f_k$$ represent the constraints that we would
need any graph sampled from the optimal probability distribution to
satisfy. Those can be any measurable function (i.e. $$statistics$$ or
$$observables$$ in the physics terminology) from the space of graphs to
$$\mathbb{R}$$ (or more generally any field).

Since the problem is convex, a general form of the solution can be
derived using the Lagrange multipliers method.

The Lagrangian of the problem is:

$$
\begin{aligned}
        \mathcal{L}(P, \lambda_1,...,\lambda_K, \mu) =
                            & -\sum\limits_{G \in \mathcal{G}} P(G) log(P(G))  \\
                            & - \sum\limits_{k=1}^{K} \lambda_k(c_k -  \mathbb{E}_P [f_k(G)]) \\
                            & - \mu (1 -\sum\limits_{G \in \mathcal{G}} P(G))
    \end{aligned}
$$

The convexity of the problem guarantees the existence of a solution. The
Lagrange multipliers theorem states that any solution must satisfy the
following condition for certain values of
$$\lambda_1^r,...,\lambda_n^r,\lambda_1^c,..., \lambda_n^c, \mu$$ and each
$$G \in \mathcal{G}$$:

$$\frac{\partial \mathcal{L}(P, \lambda_1,...,\lambda_K, \mu)}{\partial P(G)} = 0$$

This can be re-written as

$$
\begin{aligned}
    -log(P(G)) - 1  + \sum\limits_k \lambda_k f_k(G)
                   + \mu  = 0\end{aligned}
$$

By isolating $$P(G)$$ we get:

$$P(G) = \frac{exp(\sum\limits_{k=1}^K\lambda_k f_k(G))}{Z}$$

where
$$Z = \sum\limits_{G \in \mathcal{G}} exp(\sum\limits_{k=1}^K\lambda_k f_k(G))$$
is the partition function of the distribution, ensuring that it sums to
$$1$$.

The constraints are satisfied for a certain value of $$\lambda$$ if and
only if the gradient of Lagrangian is zero at these value. As a
consequence solving the (constrained) MaxEnt problem is equivalent to
solving the unconstrained optimization of the Lagrangian with respect to
$$\lambda$$. This general formula provides us with a general way of
computing a MaxEnt distribution.

Two elements need to be defined to derive a MaxEnt distribution:

- $$f_1,...,f_K$$: The statistics defining the constraints that we need
  the distribution to satisfy.

- $$\mathcal{G}$$ : the graph space that we are working on (e.g. binary
  graphs, multi graphs, weighted graphs \...). This space will be used
  to compute the partition function.

Then to fit the MaxEnt to the model, one must fit the Lagrange dual
function, which can be viewed as the negative log-likelihood of the
parameters $$\lambda_1,...,\lambda_K$$ under the observation G.

$$L(\lambda) = log(Z(\lambda)) - \sum\limits_{k=1}^K \lambda_k f_k(G)$$

# The equivalence classes trick

In order to avoid having to optimize the Lagrangian over $$n$$ variables,
in [@8543671] the authors proposes to use a trick that is inherent to
the form of the Maxent Problem.

## Introductory example

Let's consider a positive convex function
$$f:\mathbb{R}^2 \mapsto \mathbb{R_+}$$, symmetric in its two arguments
i.e. $$\forall (x,y), f(x,y) = f(y,x)$$. Suppose that we want to minimize
$$f$$.

Then the convexity of $$f$$ ensures that there exists as solution
$$(x^*,y^*)$$ of the problem is such that $$x^* = y^*$$.

Suppose we have a solution such that $$x^* \neq y^*$$. Let
$$u = \frac{1}{2}((x^*,y^*) + (y^*,x^*))$$. By convexity we have
$$f(u)\leq\frac{1}{2}(f(x^*,y^*) + f(y^*,x^*)) = f(x^*,y^*)$$ u is a
solution that verifies the condition, so the statement holds by
construction.

Note that if $$f$$ was strictly convex, we would have
$$f(u)<\frac{1}{2}(f(x^*,y^*) + f(y^*,x^*)) = f(x^*,y^*)$$ So any
solution should satisfy $$x^* = y^*$$ in the strictly convex case.

The previous situation can be formulated as follows : there is an
operation $$\tau_{12}$$ that swaps the first and second argument of the
function while leaving its value unchanged.

Let's now consider the object $$H=(\{id_2, \tau_{12}\}, *)$$ where $$id_2$$
is the identity and $$*$$ is the composition operator. This object is
called a _group_. It is actually the group of permutations of the
arguments of $$f$$ that leave its values unchanged.

The _orbit_ of $$1$$ in this group is all the elements of $$\{1,2\}$$ that
can be reached starting from 1 using permutations of the group. In this
case it's simply $$\{1, 2\}$$. A slightly more complex case would be if
$$f$$ had $$4$$ arguments $$x_1,x_2,x_3,x_4$$ and that the group was
$$H=(\{id_2, \tau_{12}, \tau_{34}\}, *)$$. In this case $$x_1$$ and $$x_2$$
can be exchanged, as well as $$x_3$$ and $$x_4$$. In this case there would
be two distinct orbits : $$\{1,2\}$$ and $$\{3,4\}$$.

## General Formulation

Let's first define some notions to formalize the idea above.

Let's denote $$\mathcal{F}$$ the set of all functions from $$\mathbb{R}^n$$
to $$\mathbb{R_+}$$, $$S_n$$ the group of permutations of $$\{1,...,n\}$$.

For each element $$f\in \mathcal{F}$$ and each $$\sigma \in S_n$$ we denote
$$f_{\sigma} : (x_1,...,x_n) \mapsto f(x_{\sigma(1)},...,x_{\sigma(n)})$$
the function $$f$$ applied to permuted arguments according to $$\sigma$$.

The **action** of the group $$S_n$$ on $$\mathcal{F}$$ is the application

$$
\begin{aligned}
F:S_n\times \mathcal{F} &\rightarrow  \mathcal{F} \\
\sigma &\mapsto f_{\sigma}    \end{aligned}
$$

For a function $$f\in \mathcal{F}$$, the **stabilizer** of $$f$$ is the set
$$Stab_f = \{\sigma \in S_n| f_\sigma = f\}$$ This set, endowed with the
composition operator, is a sub-group of $$(S_n, *)$$. It contains the
permutations that, applied to the arguments of $$f$$, leave its value
invariant.

In general $$Stab_f$$ is strictly included in $$S_n$$. In the particular
case where $$Stab_f = S_n$$, $$f$$ is called a symmetric function.

For any subgroup $$G$$ of $$S_n$$, acting of the set of indices
$$\{1,...,n\}$$ and any $$i\in \{1,...,n\}$$, the **orbit** of $$i$$ in $$G$$ is
the set $$orb_G(i) = \{j| \exists \sigma \in G, j=\sigma(i)\}$$

The collection of all the orbits form a partition of $$\{1,...,n\}$$

The general theorem is the following:

Let $$f : \mathbb{R}^n \mapsto \mathbb{R_+}$$ be a positive convex
function that we seek to minimize.

Let $$Stab_f$$ be the stabilizer of $$f$$, and $$orb_{G_f}(i)$$ denote the
orbit of $$i$$ in $$Stab_f$$, as defined above.

_Then_ $$\forall i = 1,...,n$$, there exists $$(x_1^*,...,x_n^*)$$ such that
$$\forall j \in orb_{Stab_f}(i), x_j^* = x_i^*$$

Moreover, if $$f$$ is strictly convex, any solution $$(x_1^*,...,x_n^*)$$
must satisfy
$$\forall i = 1,...,n, \forall j \in orb_{Stab_f}(i), x_j^* = x_i^*$$

Concretely, if $$f$$ is a strictly convex function of $$n$$ variables
$$x_1, ..., x_n$$, such that some subset of variables can be exchanged
without changing the value of the function, then the value of the
variables within each subset should be equal at the optimum. As a
consequence the optimization of $$f$$ can be done by optimizing an annex
function $$\tilde{f}$$ that has as many parameters as there are subsets
(i.e. orbits of the stabilizer of f).

## Application to Maxent

In the case of MaxEnt, this trick can be applied for the constraints
statistics of the type
$$f_k(G) = \sum\limits_{ij} \mathbf{1}_{\{(i,j)\in S_k\}}a_{ij}$$ for some
constraint sets $$S_1,...,S_K$$.

Let's define for any $$k$$ the feature matrix
$$F^{(k)} = (f_{ij}^{(k)})_{i,j} = (\mathbbm{1}_{\{(i,j) \in S_k\}})_{i,j}.$$

For any $$i,j$$ one can also define the feature vector
$$f_{ij} = (f_{ij}^1,...,f_{ij}^K)$$

Indeed in these cases the Lagrange dual can be written
$$\mathcal{L}(\lambda) = A(\lambda) - H(\lambda)$$ where $$\begin{aligned}
        A(\lambda) =& \sum\limits_{ij} log(1 + exp(\sum_{k=1}^K \lambda_k \mathbbm{1}_{\{(ij) \in S_k\}}) \\
        =& \sum\limits_{ij} log(1 + exp(\lambda ^T f_{ij}))\end{aligned}$$
is the log partition function and

$$
\begin{aligned}
    H(\lambda) =& \sum\limits_{k=1}^K \lambda_k  (\sum\limits_{ij} a_{ij} \mathbbm{1}_{\{(ij) \in S_k\}}) \\
    =& \sum\limits_{k=1}^K \lambda_k tr(A^t F^{(k)}) \\
    =& \sum\limits_{k=1}^K \lambda_k d_k \\\end{aligned}
$$

is the Hamiltonian. (here $$tr$$ denotes the trace operation and
$$d_k = tr(A^t F^{(k)})$$)

We can see that any permutation that leave invariant $$A$$ and $$H$$ will
leave invariant the Lagrangian. In other words
$$Stab_{A} + Stab_H \subset Stab_{\mathcal{L}}$$. So a way to finding
permutations that leave invariant the Lagrangian is find permutation
that are in both stabilizers, or in other terms to determine the direct
sum of these two stabilizers. In the derivations below we give a way to
do this.

### Stabilizer of the log-partition function

Here we give a way of finding permutations of the arguments that leave
the log-partition function invariant.

Let's consider a collection of constraint sets $$S_1,...,S_K$$.

There exists a partition $$D_1,...,D_q$$ of $$\{1,...,Q\}$$ such that:
$$\forall q=1,...,Q, \forall (k,l) \in D_q^2,  S_k \bigcap S_l = \o$$ In
other terms there is a partition of this collection such that all the
sets within an family in of sets that partition are disjoint.

The stabilizer of the log-partition function is isomorphic to the direct
sum $$\bigoplus\limits_{q=1}^Q S_{D_q}$$ where $$S_D$$ is denotes the group
of all the permutations of the elements of $$D$$.

In practice, this means that we can subdivide the constraints into
_constraint types_, with all the constraints set within a single
constraint type being disjoint.

In the case of the row and column degree, the constraints can be
subdivided into row degree constraints and column degree constraints.

### Stabilizer of the Hamiltonian

Let's decompose the function $$H$$ using the partition into constraint
types.

$$
\begin{aligned}
H(\lambda) =& \sum\limits_{k=1}^K \lambda_k d_k \\
=& \sum\limits_{q=1}^Q\sum\limits_{k\in D_Q} \lambda_k  d_k \\\end{aligned}
$$

Let's assume for all the $$q=1,...,Q$$ that for any $$k$$ in $$D_q$$ the
$$d_k^{(q)}$$ take values in the set
$$\{\delta_1^{(q)},...,\delta_{n_q}^{(q)}\}$$.

For $$k'=1,...,n_q$$, let's write $$E_{k'}^{(q)}$$ the set
$$\{k\in D_q|d_k=\delta_{k'}^{(q)}\}$$

We can rewrite $$H$$ as

$$
\begin{aligned}
H(\lambda) =& \sum\limits_{q=1}^Q \sum\limits_{k'=1}^{n_q} \delta_{k'}^{(q)} \sum\limits_{k \in E_{k'}^{(q)}} \lambda_k\\\end{aligned}
$$

In this expression it can be seen that the $$\lambda_k$$ within a single
$$E_{k'}^{(q)}$$ can be exchanged without changing the value of $$H$$.

Thus one can show that the stabilizer of $$H$$ is, up to an isomorphism
equal to the following direct sum:
$$Stab_H = \bigoplus\limits_{q=1}^Q \bigoplus\limits_{k'=1}^{n_q} S_{E_{k'}^{(q)}}$$

### Example: Row and Column degree constraints

In the case of the Row and Degree constraints we have

- $$A(\lambda) = \sum\limits_{ij} log(1 + exp(\lambda_i^r + \lambda_j^c))$$

- $$H(\lambda) = \sum\limits_{i=1}^n \lambda_i^r d_i^r + \sum\limits_{j=1}^n \lambda_j^c d_j^c$$

First we look at the stabilizer of $$A$$. We can see that swapping any row
or column indices in the log partition function doesn't change the value
of the total sum. So the _stabilizer_ of $$A$$ is actually $$S_n+S_n$$,
where $$+$$ denotes the _direct_ sum operation.

Let's now look at the Hamiltonian. Lets assume that the $$d_i^r$$ and the
$$d_j^c$$ take they values in the sets $$\{\delta_1^r,...,\delta_R^r\}$$ and
$$\{\delta_1^c,...,\delta_C^c\}$$ respectively. $$R$$ and $$C$$ represent the
number of distinct values of degrees for the rows and columns
respectively. For each $$k=1,...,R$$ and $$l=1,...,C$$ let's denote $$n_k$$
and $$m_l$$ the frequency of each row and column degree value
respectively. The Hamiltonian thus simplifies into $$\begin{aligned}
H(\lambda) =& \sum\limits_{i=1}^n \lambda_i^r d_i^r + \sum\limits_{j=1}^n \lambda_j^r d_j^c \\
=& \sum\limits_{k=1}^R \delta_k^r \sum\limits_{i=1}^n \lambda_i^r \mathbbm{1}_{\{d_i=\delta_k\}} + \sum\limits_{l=1}^C \delta_l^c \sum\limits_{j=1}^n \lambda_j^r \mathbbm{1}_{\{d_j=\delta_l\}} \\
=& \sum\limits_{k=1}^R \delta_k^r \sum\limits_{i\in E_k^r} \lambda_i^r + \sum\limits_{l=1}^C \delta_l^c \sum\limits_{j\in E_l^r} \lambda_j^r\end{aligned}$$

Where $$E_k^r = \{i|d_i^r=\delta_k^r\}$$ and
$$E_l^r = \{j|d_j^c=\delta_l^c\}$$

In this last expression it can be seen that the lambdas having their
index within a single $$E_k^r$$ or $$E_j^r$$ can be exchanged without
changing the value of H.

For each $$k$$ and $$l$$ let's denote $$G_k^r$$ (resp. $$G_k^r$$) the group of
all the permutations of the elements of $$E_k^r$$ (resp.$$E_l^c$$).

Each $$G_k^r$$ is a subgroup of $$S_n$$ so the direct sum
$$G_{1:R}=G_1^r + ... + G_R^r$$ is also a subgroup.

The stabilizer of the Hamiltonian can thus be written as
$$G_H = G_1^r + ... + G_R^r + G_1^c + ... + G_C^c$$

In order words the sets are precisely the orbits of the stabilizer of
$$H$$.

In this context the general theorem stated before can be applied.

# Bernoulli

## Directed graph

In this section we make all the derivations above in the case where
$$\mathcal{G}$$ is the set of _binary, directed graph_, and where the
constraints are the out-degrees and in-degrees of each node. We also
prove that the maximum entropy distribution under degree constraints is
an independent product of the distributions for each node pair.

Let $$\mathcal{G}$$ be the set of all possible directed binary graphs
without self-loops, on a given set of nodes $$1,...,n$$, and
$$\mathcal{P(G)}$$ the set of all probability distributions over
$$\mathcal{G}$$. A practical, algebraic way of describing the set
$$\mathcal{G}$$ is by considering the set of adjacency matrices
$$\mathcal{A} = \{(a_{ij}) \in \{0,1\} ^{n^2} | \forall i,  a_{ii} = 0\}$$.
In what follows we will combine these two notations.

The Maxent convex optimization problem consists of finding the element
of $$\mathcal{P}$$ that has maximal entropy.

It can be written as:

$$
\left\{
\begin{array}{cc}
        \max\limits_P -\sum\limits_{G \in \mathcal{G}} P(G) log(P(G)) \\
        s.t. \left\{
        \begin{array}{cc}
              \forall i,  \mathbb{E}_P [\sum\limits_{j} a_{i,j}] = d_i \\
              \forall j,  \mathbb{E}_P [\sum\limits_{i} a_{i,j}] = d_j  \\
              \sum\limits_{G \in \mathcal{G}} P(G) = 1
        \end{array}

        \right.
\end{array}
\right.
$$

Where $$\mathbb{E}_P$$ denotes the expectation under the probability $$P$$,
and the $$a_{ij}$$ denote the coefficients of the adjacency matrix (not
necessarily symmetric).

By introducing the Lagrange multipliers $$\lambda_1^r,...,\lambda_n^r$$
for the out degree (row sums) constraints,
$$\lambda_1^c,..., \lambda_n^c$$ for the in-degree (column sums)
constraints, and $$\mu$$ for the normalization constraint, the Lagrangian
of this optimization problem writes :

$$
\begin{aligned}
    \mathcal{L}(P, \lambda_1^r,...,\lambda_n^r,\lambda_1^c,..., \lambda_n^c, \mu) =
                        & -\sum\limits_{G \in \mathcal{G}} P(G) log(P(G))  \\
                        & - \sum\limits_i \lambda_i^r(d_i - \sum\limits_{j} \mathbb{E}_P [a_{i,j}]) \\
                        & - \sum\limits_j \lambda_j^c(d_j - \sum\limits_{i} \mathbb{E}_P [a_{i,j}] \\
                        & - \mu (1 -\sum\limits_{G \in \mathcal{G}} P(G))\end{aligned}
$$

At the optimum the solution must satisfy the following condition:

$$\frac{\partial \mathcal{L}(P, \lambda_1^r,...,\lambda_n^r,\lambda_1^c,..., \lambda_n^c, \mu)}{\partial P(G)} = 0$$

Let's also recall that
$$\mathbb{E}_{P}[a_{ij}] = \sum\limits_{G \in \mathcal{G}} P(G) a_{ij}$$
This yields the following equation:

$$
\begin{aligned}
    -log(P(G)) - 1  + \sum\limits_i \lambda_i^r(\sum\limits_{j} a_{i,j})
                   + \sum\limits_j \lambda_j^c(\sum\limits_{i} a_{i,j})
                   + \mu  = 0\end{aligned}
$$

Let $$Z = exp(1 - \mu)$$ denote a positive normalization constant, the
optimal probability distribution can be written as:

$$
\begin{aligned}
    P(G) = & \frac{exp(\sum\limits_i \lambda_i^r(\sum\limits_{j} a_{i,j})
                       + \sum\limits_j \lambda_j^c(\sum\limits_{i} a_{i,j}))}{Z} \\
         =  & \frac{exp(\sum\limits_{ij}(\lambda_i^r + \lambda_i^c) a_{ij})}{Z} \end{aligned}
$$

$$Z$$ is the partitions function of the distribution. Using the
normalization condition we get

$$
\begin{aligned}
    Z = &\sum\limits_{G \in \mathcal{G}} exp(\sum\limits_{ij} (\lambda_i^r + \lambda_j^c) a_{ij}) \\
      = &\sum\limits_{G \in \mathcal{G}} \prod_{ij}exp((\lambda_i^r + \lambda_j^c) a_{ij})\\
      = &\prod_{ij} \sum\limits_{a_{ij} \in \{0,1\}} exp((\lambda_i^r + \lambda_j^c) a_{ij})\end{aligned}
$$

The last equality comes from the following algebraic result:

Let $$\mathcal{S}$$ be a measurable space, $$n \geq 1$$ and
$$\theta_1,...,\theta_n$$ some real numbers such that

$$\forall i=1,...,n, \int_{s \in S} exp(\theta_i s) ds < +\infty$$

then the following statements hold :

$$\int_{s_1,...,s_n \in S^n} exp(\sum\limits_{i=1}^n \theta_i s) ds < +\infty$$
and

$$\int_{s_1,...,s_n \in S^n} exp(\sum\limits_{i=1}^n \theta_i s_i) ds_1...ds_n = \prod\limits_{i=1}^n \int_{s \in S} exp(\theta_i s)ds$$

In particular with $$S = \{0,1\}$$ we have: $$\begin{aligned}
\sum\limits_{(u_1...u_n) \in \{0;1\}^n} exp(-\sum\limits_{i} \theta_i u_i)
&= \prod_{i \in {1...n}} (\sum\limits_{u_i \in \{0;1\}} exp(-\theta_i u_i))\end{aligned}$$

Using the properties of the integral:

$$
\begin{aligned}
\int_{s_1,...,s_n \in S^n} exp(\sum\limits_{i=1}^n \theta_i s_i) ds_1...ds_n =& \int_{s_1\in S}...\int_{s_n \in S} exp(\sum\limits_{i=1}^n \theta_i s_i) ds_1...ds_n \\
&= \int_{s_1\in S}...\int_{s_n \in S} \prod\limits_{i=1}^n exp(\theta_i s_i) ds_1...ds_n \\
&= (\int_{s_1\in S}exp(\theta_1 s_1) ds_1) ... (\int_{s_n \in S} exp(\theta_n s_n) ds_n) \end{aligned}
$$

Hence the result.

Applying this result to the arrays
$$(a_{ij})_{1 \leq i,j \leq n} \in \mathcal{A}$$, and with the
coefficients $$\theta_{ij} = \lambda_i^r + \lambda_j^c$$ yields the
desired equality.

Combining all the previous expressions leads us to the final
distribution:

$$
\begin{aligned}
    P(G) =  & \prod_{ij}\frac{exp((\lambda_i^r + \lambda_j^c) a_{ij})}{1 + exp(\lambda_i^r + \lambda_j^c)} \end{aligned}
$$

This shows that without any assumption on the independence of the
presence/absence of edges between nodes in the graph, the final
probability distribution is a product of independent distributions for
each node couple.

After reducing the number of variables using the equivalent variables
trick, we get the reduced objective function $$L(\nu)$$.

The results are summarized in table
[\[table1\]](#table1){reference-type="ref" reference="table1"}:

\resizebox{\textwidth}{!}{
\begin{tabular}{|c|c|}
\hline
\centering
MaxEnt Distribution & $$P(G) = \prod\limits_{ij} \frac{exp((\lambda_i + \lambda_j) a_{ij})}{1 + exp(\lambda_i + \lambda_j)}$$ \\
\hline
\centering
Lagrange dual & $$\mathcal{L}(\lambda) = \sum\limits_{ij} log(1 + exp(\lambda_i^r + \lambda_j^c)) - \sum\limits_{i=1}^n \lambda_i^r d_i^r - \sum\limits_{j=1}^n \lambda_j^c d_j^c$$\\

    \hline
    \centering
        Reduced Objective function & $$L(\nu) = \sum\limits_{k,l} n_k m_l log(1 + exp(\nu_i^r + \nu_j^c))
    - \sum\limits_{k} n_k \nu_k^r \delta_k^r - \sum\limits_{l} m_l \nu_l^c \delta_l^c$$
        \\
    \hline

    \centering
        Derivatives wrt. $$\nu_k^r$$& $$\frac{\partial L(\nu)}{\partial \nu_k^r} = \sum\limits_{l \neq k} n_k m_l \frac{exp(\nu_k^r + \nu_l^c)}{1 + exp(\nu_k^r + \nu_l^c)} - n_k \delta_k^r  $$ \\
    \hline

    \centering
        Derivatives wrt. $$\nu_l^c$$& $$\frac{\partial L(\nu)}{\partial \nu_l^c} = \sum\limits_{k \neq l} n_k m_l \frac{exp(\nu_k^r + \nu_l^c)}{1 + exp(\nu_k^r + \nu_l^c)} - n_l \delta_l^c  $$ \\

    \hline
    \end{tabular}

}
Undirected Graph

---

Here let's consider the case where $$\mathcal{G}$$ is the set of _binary,
undirected_ graphs. Equivalently the set of adjacency matrices we're
interested in is the set
$$\mathcal{A} = \{(a_{ij}) \in \{0,1\} ^{n^2} | \forall {i,j}, a_{ij} = a_{ji}; a_{ii} = 0\}$$.

The MaxEnt problem is similar to the one in the oriented case, but with
only one constraint per node: $$\left\{
\begin{array}{cc}
        \max\limits_P -\sum\limits_{G \in \mathcal{G}} P(G) log(P(G)) \\
        s.t. \left\{
        \begin{array}{cc}
              \forall i, \mathbb{E}_P [\sum\limits_{j=1}^n a_{ij}] = d_i \\
              \sum\limits_{G \in \mathcal{G}} P(G) = 1
        \end{array}
        \right.
\end{array}
\right.$$

The solution of this problem is of the form :

$$
\begin{aligned}
P(G) =& \frac{exp(\sum\limits_{i=1}^n\lambda_i (\sum\limits_{j=1}^n a_{ij}))}{Z} \end{aligned}
$$

\resizebox{\textwidth}{!}{
\begin{tabular}{|c|c|}
\hline
MaxEnt Distribution & $$P(G) = \prod_{i<j} \frac{exp((\lambda_i + \lambda_j) a_{ij})}{1 + exp(\lambda_i + \lambda_j)}$$ \\
\hline
Lagrange dual & $$L(\lambda) = \sum\limits_{i<j} log(1 + exp(\lambda_i + \lambda_j)) - \sum\limits_{i=1}^n \lambda_i d_i  $$\\

    \hline
    \centering
        Reduced Objective function & $$L(\nu) = \sum\limits_{k<l} n_k n_l log(1 + exp(\nu_k + \nu_l)) - \sum\limits_{k} n_k \nu_k \delta_k  $$
        \\
    \hline

    \centering
        Derivatives & $$\frac{\partial L(\nu)}{\partial \nu_k} = \sum\limits_{l \neq k} n_k n_l \frac{exp(\nu_k + \nu_l)}{1 + exp(\nu_k + \nu_l)} - n_k \delta_k  $$
        \\
    \hline

    \centering
        Hessian for $$k = l$$ & $$\frac{\partial ^2 L(\nu)}{\partial \nu_k ^2} = \sum\limits_{l \neq k} n_k n_l \frac{exp(\nu_k + \nu_l)}{(1 + exp(\nu_k + \nu_l))^2}$$
        \\
    \hline
    \centering
        Hessian for $$k \neq l$$ & $$\frac{\partial ^2 L(\nu)}{\partial \nu_k\partial \nu_l} = n_k n_l \frac{exp(\nu_k + \nu_l)}{(1 + exp(\nu_k + \nu_l))^2} $$
        \\
    \hline
    \end{tabular}

}
Using the properties of the exponential and product we get:

$$
\begin{aligned}
    exp(\sum\limits_{i=1}^n\lambda_i (\sum\limits_{j=1}^n a_{ij})) = & \prod_{i<j} exp(\lambda_i a_{ij}) \times \prod_{i} exp(\lambda_i a_{ii}) \times \prod_{i<j} exp(\lambda_i a_{ij}) \\
    =&\prod_{i<j} exp(\lambda_i a_{ij})  \prod_{j<i} exp(\lambda_i a_{ij}) &&\text{(diagonal terms are zero)}\\
    =&\prod_{i<j} exp(\lambda_i a_{ij})  \prod_{j<i} exp(\lambda_i a_{ji}) && \text{ (since A is symmetric)}\\
    =&\prod_{i<j} exp(\lambda_i a_{ij})  \prod_{i<j} exp(\lambda_j a_{ij}) && \text{ (by swapping the indices)}\\
    =&\prod_{i<j} exp((\lambda_i + \lambda_j) a_{ij})  \end{aligned}
$$

Now let's factorize the partition function. Using Lemma 1, the partition
function is $$\begin{aligned}
    Z = &\sum\limits_{G \in \mathcal{G}} exp(\sum\limits_{i<j} (\lambda_i + \lambda_j) a_{ij}) \\
      = &\prod_{i<j} \sum\limits_{a_{ij} \in \{0,1\}} exp((\lambda_i + \lambda_j) a_{ij})\end{aligned}$$

Once again we see that the solution distribution can be expressed as a
product of independent Bernouilli distributions, one for each node pair.

## Maxent with \"Block Degree\" constraint

Similarly as in DeBayes, let's consider a specific case of MaxEnt, where
each node $$j$$ has a unique attribute $$s_j\in \{1,...,S\}$$ and the
constraint statistics are, for each node and each attribute value, the
number of edges from that node to nodes having this attribute value:

$$
\left\{
\begin{array}{cc}
        \max\limits_P -\sum\limits_{G \in \mathcal{G}} P(G) log(P(G)) \\
        s.t. \left\{
        \begin{array}{cc}
              \forall i, s, \mathbb{E}_P [\sum\limits_{j=1}^n a_{ij} \mathbbm{1}_{\{s_j = s\}}] = d_i^s \\
              \sum\limits_{G \in \mathcal{G}} P(G) = 1
        \end{array}
        \right.
\end{array}
\right.
$$

In this case the MaxEnt distribution has the form

$$P(G) = \prod\limits_{i < j} \frac{exp((\lambda_i^{s_j} + \lambda_j^{s_i}) a_{ij})}{1 + exp((\lambda_i^{s_j} + \lambda_j^{s_i}))}$$

The Lagrange dual function, expressed as a function of the parameters
$$\lambda_i^s$$ is:

$$L(\lambda) = \sum_{i<j} log(1 + exp(\lambda_i^{s_j} + \lambda_j^{s_i})) - \sum_{i=1}^n\sum_{s=1}^S \lambda_{i}^s d_{i}^s$$

The Hamiltonian term is
$$H(\lambda)=\sum_{i=1}^n\sum_{s=1}^S \lambda_{i}^s d_{i}^s$$.

For any $$s\in \{1,...,S\}$$, if two indices $$i, i'$$ have the same value
$$d_i^s = d_{i'}^s$$, swapping them in the sum will leave the Hamiltonian
value invariant.

The cumulant term is

$$
\begin{aligned}
A(\lambda)
=& \sum_{i<j} log(1 + exp(\lambda_i^{s_j} + \lambda_j^{s_i})) \\
=& \sum_{i<j} \sum_{s, t} \mathbbm{1}_{\{s_j=s, s_i=t\}}log(1 + exp(\lambda_i^{s} + \lambda_j^{t})) \\    \end{aligned}
$$

Let's suppose that for each $$i, s$$, the degrees $$d_i^s$$ take only a
number $$N_s$$ of distinct values $$\delta_1^s,...,\delta_{K_s}^s$$.

Let's define $$n_k^s = Card\{i| d_i^s = \delta_k^s\}$$ the frequency of
the value $$\delta_k^s$$.

Let's also define
$$m_{k}^{s, t} = Card\{i| d_i^s = \delta_k^s , s_i = t\}$$ the number of
nodes in cluster $$C_t$$ that share exactly $$\delta_k^s$$ edges with nodes
in the cluster $$C_s$$.

Similarly as before, we use the equivalence class trick to reduce the
number of distinct optimization variables.

We get the following reduced objective

$$
L(\nu)
= \sum_{k<l} \sum_{s, t} m_k^{s,t} m_l^{t, s}log(1 + exp(\nu_{k}^s + \nu_l^t))
- \sum_{k}\sum_s \nu_k^s n_k^s \delta_k^s
$$

# Integer Weighted Graph

In this case we consider the set $$\mathcal{G}$$ of undirected weighted
graphs, where the weights between two edges can be any integer. The
equivalent adjacency matrix set is
$$\mathcal{A} = \{ (a_{ij} \in \mathbb{N}^n | \forall (i,j), a_{ij} = a_{ji}, a_{ii} = 0\}$$
In the case of the degree constraints, the MaxEnt problem is $$\left\{
\begin{array}{cc}
        \max\limits_P -\sum\limits_{G \in \mathcal{G}} P(G) log(P(G)) \\
        s.t. \left\{
        \begin{array}{cc}
              \forall i, \mathbb{E}_P [\sum\limits_{j=1}^n a_{ij}] = d_i \\
              \sum\limits_{G \in \mathcal{G}} P(G) = 1
        \end{array}
        \right.
\end{array}
\right.$$

and the resulting MaxEnt distribution has the form

$$P(G) = \frac{\prod\limits_{i<j} exp((\lambda_i + \lambda_j) a_{ij})}{Z}$$

We can again use Lemma 1 to compute the partition function, with this
time the space $$\mathbb{S}$$ being equal to $$\mathbb{N}$$.

To fill in the conditions of the lemma, the constraints
$$\lambda_i + \lambda_j < 0$$ must be satisfied for all $$1<i<j<n$$, when
fitting the MaxEnt model to an observed graph.

$$
\begin{aligned}
    Z =& \sum\limits_{G \in \mathcal{G}} \prod\limits_{i<j} exp((\lambda_i + \lambda_j) a_{ij}) \\
    =& \prod\limits_{i<j}  \sum\limits_{a_{ij} \in \mathbb{N}} exp((\lambda_i + \lambda_j) a_{ij}) \\
    =& \prod\limits_{i<j}  \frac{1}{1 - exp(\lambda_i + \lambda_j)} \\\end{aligned}
$$

So finally the resulting distribution is thus a product of independent
Geometric distributions with parameter $$exp(\lambda_i + \lambda_j)$$ for
each node pair $$i,j$$

# Graph with positive real valued weights

In this case we consider the set $$\mathcal{G}$$ of undirected weighted
graphs, where the weights between two edges can be positive real number.
The equivalent adjacency matrix set is
$$\mathcal{A} = \{ (a_{ij} \in (\mathbb{R}_+)^n | \forall (i,j), a_{ij} = a_{ji}, a_{ii} = 0\}$$
In the case of the degree constraints, the MaxEnt problem is $$\left\{
\begin{array}{cc}
        \max\limits_P -\sum\limits_{G \in \mathcal{G}} P(G) log(P(G)) \\
        s.t. \left\{
        \begin{array}{cc}
              \forall i, \mathbb{E}_P [\sum\limits_{j=1}^n a_{ij}] = d_i \\
              \sum\limits_{G \in \mathcal{G}} P(G) = 1
        \end{array}
        \right.
\end{array}
\right.$$

and the resulting MaxEnt distribution has the form

$$P(G) = \frac{\prod\limits_{i<j} exp((\lambda_i + \lambda_j) a_{ij})}{Z}$$

We can again use Lemma 1 to compute the partition function, with this
time the space $$\mathbb{S}$$ being equal to $$\mathbb{N}$$.

To fill in the conditions of the lemma, the constraints
$$\lambda_i + \lambda_j < 0$$ must be satisfied for all $$1<i<j<n$$, when
fitting the MaxEnt model to an observed graph.

$$
\begin{aligned}
    Z =& \int_{G \in \mathcal{G}} \prod\limits_{i<j} exp((\lambda_i + \lambda_j) a_{ij}) \\
    =& \prod\limits_{i<j}  \int_{a_{ij}=0}^{+\infty} exp((\lambda_i + \lambda_j) a_{ij}) \\
    =& \prod\limits_{i<j}  \frac{- 1}{\lambda_i + \lambda_j} \\\end{aligned}
$$

So finally the resulting distribution is thus a product of independent
Geometric distributions with parameter $$exp(\lambda_i + \lambda_j)$$ for
each node pair $$i,j$$

\resizebox{\textwidth}{!}{
\begin{tabular}{|c|c|}
\hline
MaxEnt Distribution & P(G) = \prod\limits*{i<j} - (\lambda_i + \lambda_j) exp((\lambda_i + \lambda_j)a*{ij}) \\
\hline
Lagrange dual & $$L(\lambda) = \sum\limits_{i<j} log( - (\lambda_i + \lambda_j)) - \sum\limits_{i=1}^n \lambda_i d_i $$\\

\hline
\centering
Reduced Objective function & $$L(\nu) = \sum\limits_{k<l} n_k n_l log(-(\nu_k + \nu_l)) - \sum\limits_{k} n_k \nu_k \delta_k  $$
\\
\hline

\centering
Derivatives & $$\frac{\partial L(\nu)}{\partial \nu_k} = \sum\limits_{l \neq k} - n_k n_l \frac{1}{\nu_k + \nu_l} - n_k \delta_k  $$
\\
\hline

\centering
Hessian for $$k = l$$ & $$\frac{\partial ^2 L(\nu)}{\partial \nu_k ^2} = \sum\limits_{l \neq k} n_k n_l \frac{1}{(\nu_k + \nu_l)^2}$$
\\
\hline
\centering
Hessian for $$k \neq l$$ & $$\frac{\partial ^2 L(\nu)}{\partial \nu_k\partial \nu_l} = n_k n_l \frac{1}{(\nu_k + \nu_l)^2} $$
\\
\hline
\end{tabular}
}
\printbibliography
