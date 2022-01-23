---
author:
- Alon Jacobson
date: 'January 22, 2022'
title: Paint Regression Project Mathematical Description
---

Introduction
============

The goal of my paint regression project is to try to get an
understanding of how artists paint, and also to get pretty pictures, by
mathematically modeling a simplified view of how artists paint a picture
from a reference photo. The idea is that a painting is constructed from
a sequence of brush strokes, with each brush stroke being optimal or
near optimal to get the painting as close as possible to what the artist
wants it to look like. This optimality constraint is motivated by the
fact that while computer monitors and printers are not limited in the
number of pixels or dots to construct the image, artists are, because
they cannot work nearly as fast, and hence, they will try to place brush
strokes strategically to maximize the impact.

Setup
=====

We have a source image $I$, represented as an array of pixels with $m$
rows, $n$ columns, and $c$ channels ($c$ will usually be 3 for red,
green, and blue). Color intensities are from 0 to 1. So,
$I \in \mathcal{I}$, where $\mathcal{I} = [0,1]^{m \times n \times c}$
is the set of all possible images. The program is trying to recreate $I$
by placing brush strokes on a painting one-by-one. The painting after
$t$ brush strokes is denoted $P_t \in \mathcal I$. The painting can
start out as anything, but in my program I start the painting as being
white to represent a blank canvas, so we can set $P_0$ to be the filled
with 1s.

Represent a brushstroke as a triple $B = (R, \vec C, O)$.

$R \in \mathcal{R}$ is the region in the image that the brush paints on,
where $\mathcal{R}$ is the set of all possible regions. If $R$ is
represented as a set of pixel positions, then set of all regions
$\mathcal{R}$ is the set of all sets of pixel positions, i.e., the set
of all subsets of the set of all pixel positions. Denote
$\mathcal{P} = \{1,\dots,m\} \times \{1,\dots,n\}$ as the set of all
pixel positions of the image. The set of all regions, then, is the power
set of $\mathcal{P}$, denoted $2^\mathcal{P}$, so in this case
$\mathcal{R} = 2^\mathcal{P}$.

$\vec C \in \mathcal C$ is the color of the brush, where $\mathcal C$ is
the set of all colors and $\mathcal C = [0,1]^c$.

$O \in \mathcal O$ is the opacity of the brush, where $\mathcal O$ is
the set of all opacities, and $\mathcal O = [0,1]$.

The "loss" of the painting after $t$ iterations, $\mathcal{L}(P_t)$, is
the total squared error between the painting and the image. We have a
function $\mathcal{L} : \mathcal I \to \mathbb{R}$ which gives the loss
of a painting: $$\begin{aligned}
\mathcal{L}(P) &:= \sum_{i=1}^m \sum_{j=1}^n \sum_{k=1}^c ( P(i,j,k) - I(i,j,k) )^2 \\
&= \sum_{i=1}^m \sum_{j=1}^n \underbrace{\| P(i,j) - I(i,j) \|^2}_{\ell_P(i,j)}\end{aligned}$$

With each iteration, a brush stroke is chosen to try to minimize the
loss at the next iteration.

Let $B_t = (R_t, \vec C_t, O_t)$ be the brushstroke used at time $t$.
The painting at iteration $t$ is given by: $$P_t = T_{B_t}(P_{t-1})$$
where $T_{R,\vec C,O}$ is a parameterized function that takes in an
image returns that image with a brush stroke applied to it:
$$[T_{R,\vec C,O}(P)](i,j) = \begin{cases}
b_{\vec C, O}(P(i,j))   &  (i,j) \in R \\
P(i,j) & \text{otherwise}
\end{cases}$$ where $$b_{\vec C,O}(\vec x) = O \vec C + (1-O) \vec x.$$

Now I will write change in the loss from applying a brush
$B=(R,\vec C,O)$. The derivation is very simple, but it just takes a lot
of lines. Write $P' = T_{B}(P)$ as shorthand. $$\begin{aligned}
\Delta \mathcal{L}_P(B) := \mathcal{L}(T_{B}(P)) - \mathcal{L}(P) &= \sum_{i,j} \ell_{P'}(i,j) - \mathcal{L}(P) \\
&= \sum_{(i,j) \in R} \ell_{P'}(i,j) + \sum_{(i,j) \notin R} \ell_{P'}(i,j) - \mathcal{L}(P) \\
&= \sum_{(i,j) \in R} \ell_{P'}(i,j) + \sum_{(i,j) \notin R} \| P'(i,j) - I(i,j) \|^2 - \mathcal{L}(P) \\
&= \sum_{(i,j) \in R} \ell_{P'}(i,j) + \sum_{(i,j) \notin R} \| P(i,j) - I(i,j) \|^2 - \mathcal{L}(P) \\
&= \sum_{(i,j) \in R} \ell_{P'}(i,j) + \sum_{(i,j) \notin R} \ell_{P}(i,j) - \mathcal{L}(P) \\
&= \sum_{(i,j) \in R} \ell_{P'}(i,j) + \sum_{i,j} \ell_{P}(i,j) - \sum_{(i,j) \in R} \ell_{P}(i,j) - \mathcal{L}(P) \\
&= \sum_{(i,j) \in R} \ell_{P'}(i,j) + \mathcal{L}(P) - \sum_{(i,j) \in R} \ell_{P}(i,j) - \mathcal{L}(P) \\
&= \sum_{(i,j) \in R} \ell_{P'}(i,j) - \sum_{(i,j) \in R} \ell_P(i,j) \\
&= \sum_{(i,j) \in R} \| P'(i,j) - I(i,j) \|^2 - \sum_{(i,j) \in R} \ell_{P}(i,j) \\
&= \sum_{(i,j) \in R} \| b_{\vec C,O}(P(i,j)) - I(i,j) \|^2 - \sum_{(i,j) \in R} \| P(i,j) - I(i,j) \|^2.\end{aligned}$$
This formula avoids the need to sum over the whole image, instead just
summing over the region painted on.

Suppose we have a painting $P$, and we have a set of brushes
$S \subseteq \mathcal{R} \times \mathcal C \times \mathcal O$ to choose
from. We want to choose the brush $B \in S$ that minimizes
$\mathcal{L}(T_{B}(P))$. This is equivalent to minimizing
$\Delta \mathcal{L}_P(B)$, since $\Delta \mathcal{L}_P(B)$ and
$\mathcal{L}(T_{B}(P))$ differ by the $\mathcal{L}(P)$, which is
constant for all brushes. Thus when finding brushes we will minimize
$\Delta \mathcal{L}_P$.

Choosing color and opacity given a brush region
===============================================

Suppose that we have the current painting $P$ to paint on, and we
already have a chosen region to paint on, $R$. Then we can choose the
color $\vec C$ and opacity $O$ that minimize
$\Delta \mathcal{L}_P(R,\vec C,O)$. As shown above,
$$\Delta \mathcal{L}_P(B) = \sum_{(i,j) \in R} \| b_{\vec C,O}(P(i,j)) - I(i,j) \|^2 - \sum_{(i,j) \in R} \| P(i,j) - I(i,j) \|^2$$
and the second term is constant for all $\vec C$ and $O$, so we want to
minimize the following quantity: $$\label{eq1}
\sum_{(i,j) \in R} \| b_{\vec C,O}(P(i,j)) - I(i,j) \|^2$$ And if we
write $\vec \alpha = O\vec C$ and $\beta = 1-O$, then
$$b_{\vec C,O}(\vec x) = O \vec C + (1-O) \vec x = \vec \alpha + \beta \vec x$$
and it doesn't matter if we optimize $\vec \alpha$ and $\beta$ instead
of $\vec C$ and $O$ because you can go back and forth between the two
forms. If we write $R = \{ (i_1,j_1), \dots, (i_N,j_N) \}$, and then
write $$X := \begin{bmatrix}
P(i_1,j_1) \\
\vdots \\
P(i_N,j_N)
\end{bmatrix}, \quad
Y := \begin{bmatrix}
I(i_1,j_1) \\
\vdots \\
I(i_N,j_N)
\end{bmatrix},$$ then we can rewrite
([\[eq1\]](#eq1){reference-type="ref" reference="eq1"}):
$$\sum_{(i,j) \in R} \| b_{\vec C,O}(P(i,j)) - I(i,j) \|^2
=
\sum_{i=1}^N \| \vec \alpha + \beta X_i - Y_i \|^2
=
\sum_{i=1}^N \sum_{j=1}^c [\vec \alpha_j + \beta X_{i,j} - Y_{i,j} ]^2.$$
We want to choose $\vec \alpha$ and $\beta$ that minimize this quantity.
By taking partial derivatives, one can obtain the following solution
$(\vec \alpha^*,\beta^*)$.

Write the mean of the $k$ component of $X$ and $Y$ as
$$\overline{X_{:,k}} := \frac{1}{N} \sum_{i=1}^N X_{i,k}, \quad
\overline{Y_{:,k}} := \frac{1}{N} \sum_{i=1}^N Y_{i,k},$$ and $S_{xx}$
and $S_{xy}$ are
$$S_{xx} := \sum_{i=1}^N \sum_{j=1}^c (X_{i,j} - \overline{X_{:,j}})^2, \quad
S_{xy} := \sum_{i=1}^N \sum_{j=1}^c (X_{i,j} - \overline{X_{:,j}}) (Y_{i,j} - \overline{Y_{:,j}})$$
then the solution is:
$$\alpha^*_k = \overline{Y_{:,k}} - \beta^* \overline{X_{:,k}},$$ and
$$\beta^* = \frac{S_{xy}}{S_{xx}} \quad \text{if } S_{xx} \neq 0.$$ If
$S_{xx} = 0$, then it must be the case that $S_{xy} = 0$ in which case
any value of $\beta^*$ works (so if $S_{xx} = 0$ but $S_{xy} \neq 0$,
then there was an arithmetic error).

Once we obtain $(\vec \alpha^*, \beta^*)$ which are optimal, we can set
$O^* = 1 - \beta^*$, $\vec C^* = \frac{\vec \alpha^*}{O^*}$. If
$O^* \notin [0,1]$ or $\vec C^* \notin [0,1]^c$, then we could try to do
a constrained optimization, but I'm lazy so in this case let's just
reject the region $R$ and pick another region.

Choosing a brush region
=======================

Now that we are able to pick the optimal brush color and opacity given a
brush region, we are concerned with picking the best brush region. There
are way too many regions to check by brute force, and you can't
differentiate with respect to the region to try to do gradient based
optimization, so what we will use random-restart hill climbing.

To generate random neighbors, "parameterized brush regions" will be
used. We have a number of parameterized brush region classes, labelled 1
through $K$. Each parameterized brush region class $k \in \{1,\dots,K\}$
has associated with it a set of possible parameters $\Theta_k$ (usually
$\Theta_k \subseteq \mathbb{R}^p$ for some $p \in \mathbb{N}$), and a
brush in that class is identified by its parameters. The class also has
a random variable $G_k : \Omega_k \to \Theta_k$ which generates random
parameters and is to be used by random search and random-restart hill
climbing; a function $r_k : \Theta_k \to \mathcal{R}$ which returns the
region of a brush given its parameters; and a function
$n_k : \Theta_k \to \Theta_k$ which generates a "neighbor" of a
parameterized region and is to be used by hill climbing.

Let's make a function that given a region and a painting, gives the
optimal color and opacity:
$p : \mathcal R \times \mathcal I \to \mathcal C \times \mathcal O$ and
$p(R, P) = (\vec C^*, O^*)$.

Algorithm
=========

The basic algorithm is as follows:

1.  Start with $P_0$ having all ones (white).

2.  For iteration $t=0$ to max iter:

    1.  Set
        $L_{\text{best}} = 0, (k_{\text{best}}, \theta_{\text{best}}) = None$

    2.  For each brush class $k=1$ through $K$:

    3.  1.  $(\theta, L) := \textsc{HillClimbing}(\lambda \theta . \Delta \mathcal{L}_{P_k}(r_k(\theta), p(r_k(\theta), P_t)), G_k, n_k)$

        2.  If $L < L_{\text{best}}$, then

            -   $(k_\text{best},\theta_\text{best}) := (k, \theta)$

            -   $L_\text{best} := L$

    4.  $P_{t+1} := T_{r_{k_\text{best}}(\theta_\text{best})}(P_t)$
