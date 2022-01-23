Paint Regression
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

I tried to convert LaTeX to Markdown so that you can see my stuff in the README file on GitHub, and I partially succeeded, but one section just didn't render properly so I had to remove it. Please refer to `paint_regression_description.pdf`.

Setup
=====

We have a source image <img src="https://i.upmath.me/svg/I" alt="I" />, represented as an array of pixels with <img src="https://i.upmath.me/svg/m" alt="m" />
rows, <img src="https://i.upmath.me/svg/n" alt="n" /> columns, and <img src="https://i.upmath.me/svg/c" alt="c" /> channels (<img src="https://i.upmath.me/svg/c" alt="c" /> will usually be 3 for red,
green, and blue). Color intensities are from 0 to 1. So,
<img src="https://i.upmath.me/svg/I%20%5Cin%20%5Cmathcal%7BI%7D" alt="I \in \mathcal{I}" />, where <img src="https://i.upmath.me/svg/%5Cmathcal%7BI%7D%20%3D%20%5B0%2C1%5D%5E%7Bm%20%5Ctimes%20n%20%5Ctimes%20c%7D" alt="\mathcal{I} = [0,1]^{m \times n \times c}" />
is the set of all possible images. The program is trying to recreate <img src="https://i.upmath.me/svg/I" alt="I" />
by placing brush strokes on a painting one-by-one. The painting after
<img src="https://i.upmath.me/svg/t" alt="t" /> brush strokes is denoted <img src="https://i.upmath.me/svg/P_t%20%5Cin%20%5Cmathcal%20I" alt="P_t \in \mathcal I" />. The painting can
start out as anything, but in my program I start the painting as being
white to represent a blank canvas, so we can set <img src="https://i.upmath.me/svg/P_0" alt="P_0" /> to be the filled
with 1s.

Represent a brushstroke as a triple <img src="https://i.upmath.me/svg/B%20%3D%20(R%2C%20%5Cvec%20C%2C%20O)" alt="B = (R, \vec C, O)" />.

<img src="https://i.upmath.me/svg/R%20%5Cin%20%5Cmathcal%7BR%7D" alt="R \in \mathcal{R}" /> is the region in the image that the brush paints on,
where <img src="https://i.upmath.me/svg/%5Cmathcal%7BR%7D" alt="\mathcal{R}" /> is the set of all possible regions. If <img src="https://i.upmath.me/svg/R" alt="R" /> is
represented as a set of pixel positions, then set of all regions
<img src="https://i.upmath.me/svg/%5Cmathcal%7BR%7D" alt="\mathcal{R}" /> is the set of all sets of pixel positions, i.e., the set
of all subsets of the set of all pixel positions. Denote
<img src="https://i.upmath.me/svg/%5Cmathcal%7BP%7D%20%3D%20%5C%7B1%2C%5Cdots%2Cm%5C%7D%20%5Ctimes%20%5C%7B1%2C%5Cdots%2Cn%5C%7D" alt="\mathcal{P} = \{1,\dots,m\} \times \{1,\dots,n\}" /> as the set of all
pixel positions of the image. The set of all regions, then, is the power
set of <img src="https://i.upmath.me/svg/%5Cmathcal%7BP%7D" alt="\mathcal{P}" />, denoted <img src="https://i.upmath.me/svg/2%5E%5Cmathcal%7BP%7D" alt="2^\mathcal{P}" />, so in this case
<img src="https://i.upmath.me/svg/%5Cmathcal%7BR%7D%20%3D%202%5E%5Cmathcal%7BP%7D" alt="\mathcal{R} = 2^\mathcal{P}" />.

<img src="https://i.upmath.me/svg/%5Cvec%20C%20%5Cin%20%5Cmathcal%20C" alt="\vec C \in \mathcal C" /> is the color of the brush, where <img src="https://i.upmath.me/svg/%5Cmathcal%20C" alt="\mathcal C" /> is
the set of all colors and <img src="https://i.upmath.me/svg/%5Cmathcal%20C%20%3D%20%5B0%2C1%5D%5Ec" alt="\mathcal C = [0,1]^c" />.

<img src="https://i.upmath.me/svg/O%20%5Cin%20%5Cmathcal%20O" alt="O \in \mathcal O" /> is the opacity of the brush, where <img src="https://i.upmath.me/svg/%5Cmathcal%20O" alt="\mathcal O" /> is
the set of all opacities, and <img src="https://i.upmath.me/svg/%5Cmathcal%20O%20%3D%20%5B0%2C1%5D" alt="\mathcal O = [0,1]" />.

The "loss" of the painting after <img src="https://i.upmath.me/svg/t" alt="t" /> iterations, <img src="https://i.upmath.me/svg/%5Cmathcal%7BL%7D(P_t)" alt="\mathcal{L}(P_t)" />, is
the total squared error between the painting and the image. We have a
function <img src="https://i.upmath.me/svg/%5Cmathcal%7BL%7D%20%3A%20%5Cmathcal%20I%20%5Cto%20%5Cmathbb%7BR%7D" alt="\mathcal{L} : \mathcal I \to \mathbb{R}" /> which gives the loss
of a painting: <img src="https://i.upmath.me/svg/%5Cbegin%7Baligned%7D%0A%5Cmathcal%7BL%7D(P)%20%26%3A%3D%20%5Csum_%7Bi%3D1%7D%5Em%20%5Csum_%7Bj%3D1%7D%5En%20%5Csum_%7Bk%3D1%7D%5Ec%20(%20P(i%2Cj%2Ck)%20-%20I(i%2Cj%2Ck)%20)%5E2%20%5C%5C%0A%26%3D%20%5Csum_%7Bi%3D1%7D%5Em%20%5Csum_%7Bj%3D1%7D%5En%20%5Cunderbrace%7B%5C%7C%20P(i%2Cj)%20-%20I(i%2Cj)%20%5C%7C%5E2%7D_%7B%5Cell_P(i%2Cj)%7D%5Cend%7Baligned%7D" alt="\begin{aligned}
\mathcal{L}(P) &amp;:= \sum_{i=1}^m \sum_{j=1}^n \sum_{k=1}^c ( P(i,j,k) - I(i,j,k) )^2 \\
&amp;= \sum_{i=1}^m \sum_{j=1}^n \underbrace{\| P(i,j) - I(i,j) \|^2}_{\ell_P(i,j)}\end{aligned}" />

With each iteration, a brush stroke is chosen to try to minimize the
loss at the next iteration.

Let <img src="https://i.upmath.me/svg/B_t%20%3D%20(R_t%2C%20%5Cvec%20C_t%2C%20O_t)" alt="B_t = (R_t, \vec C_t, O_t)" /> be the brushstroke used at time <img src="https://i.upmath.me/svg/t" alt="t" />.
The painting at iteration <img src="https://i.upmath.me/svg/t" alt="t" /> is given by: <img src="https://i.upmath.me/svg/P_t%20%3D%20T_%7BB_t%7D(P_%7Bt-1%7D)" alt="P_t = T_{B_t}(P_{t-1})" />
where <img src="https://i.upmath.me/svg/T_%7BR%2C%5Cvec%20C%2CO%7D" alt="T_{R,\vec C,O}" /> is a parameterized function that takes in an
image returns that image with a brush stroke applied to it:
<img src="https://i.upmath.me/svg/%5BT_%7BR%2C%5Cvec%20C%2CO%7D(P)%5D(i%2Cj)%20%3D%20%5Cbegin%7Bcases%7D%0Ab_%7B%5Cvec%20C%2C%20O%7D(P(i%2Cj))%20%20%20%26%20%20(i%2Cj)%20%5Cin%20R%20%5C%5C%0AP(i%2Cj)%20%26%20%5Ctext%7Botherwise%7D%0A%5Cend%7Bcases%7D" alt="[T_{R,\vec C,O}(P)](i,j) = \begin{cases}
b_{\vec C, O}(P(i,j))   &amp;  (i,j) \in R \\
P(i,j) &amp; \text{otherwise}
\end{cases}" /> where <img src="https://i.upmath.me/svg/b_%7B%5Cvec%20C%2CO%7D(%5Cvec%20x)%20%3D%20O%20%5Cvec%20C%20%2B%20(1-O)%20%5Cvec%20x." alt="b_{\vec C,O}(\vec x) = O \vec C + (1-O) \vec x." />

Now I will write change in the loss from applying a brush
<img src="https://i.upmath.me/svg/B%3D(R%2C%5Cvec%20C%2CO)" alt="B=(R,\vec C,O)" />. The derivation is very simple, but it just takes a lot
of lines. Write <img src="https://i.upmath.me/svg/P'%20%3D%20T_%7BB%7D(P)" alt="P' = T_{B}(P)" /> as shorthand. <img src="https://i.upmath.me/svg/%5Cbegin%7Baligned%7D%0A%5CDelta%20%5Cmathcal%7BL%7D_P(B)%20%3A%3D%20%5Cmathcal%7BL%7D(T_%7BB%7D(P))%20-%20%5Cmathcal%7BL%7D(P)%20%26%3D%20%5Csum_%7Bi%2Cj%7D%20%5Cell_%7BP'%7D(i%2Cj)%20-%20%5Cmathcal%7BL%7D(P)%20%5C%5C%0A%26%3D%20%5Csum_%7B(i%2Cj)%20%5Cin%20R%7D%20%5Cell_%7BP'%7D(i%2Cj)%20%2B%20%5Csum_%7B(i%2Cj)%20%5Cnotin%20R%7D%20%5Cell_%7BP'%7D(i%2Cj)%20-%20%5Cmathcal%7BL%7D(P)%20%5C%5C%0A%26%3D%20%5Csum_%7B(i%2Cj)%20%5Cin%20R%7D%20%5Cell_%7BP'%7D(i%2Cj)%20%2B%20%5Csum_%7B(i%2Cj)%20%5Cnotin%20R%7D%20%5C%7C%20P'(i%2Cj)%20-%20I(i%2Cj)%20%5C%7C%5E2%20-%20%5Cmathcal%7BL%7D(P)%20%5C%5C%0A%26%3D%20%5Csum_%7B(i%2Cj)%20%5Cin%20R%7D%20%5Cell_%7BP'%7D(i%2Cj)%20%2B%20%5Csum_%7B(i%2Cj)%20%5Cnotin%20R%7D%20%5C%7C%20P(i%2Cj)%20-%20I(i%2Cj)%20%5C%7C%5E2%20-%20%5Cmathcal%7BL%7D(P)%20%5C%5C%0A%26%3D%20%5Csum_%7B(i%2Cj)%20%5Cin%20R%7D%20%5Cell_%7BP'%7D(i%2Cj)%20%2B%20%5Csum_%7B(i%2Cj)%20%5Cnotin%20R%7D%20%5Cell_%7BP%7D(i%2Cj)%20-%20%5Cmathcal%7BL%7D(P)%20%5C%5C%0A%26%3D%20%5Csum_%7B(i%2Cj)%20%5Cin%20R%7D%20%5Cell_%7BP'%7D(i%2Cj)%20%2B%20%5Csum_%7Bi%2Cj%7D%20%5Cell_%7BP%7D(i%2Cj)%20-%20%5Csum_%7B(i%2Cj)%20%5Cin%20R%7D%20%5Cell_%7BP%7D(i%2Cj)%20-%20%5Cmathcal%7BL%7D(P)%20%5C%5C%0A%26%3D%20%5Csum_%7B(i%2Cj)%20%5Cin%20R%7D%20%5Cell_%7BP'%7D(i%2Cj)%20%2B%20%5Cmathcal%7BL%7D(P)%20-%20%5Csum_%7B(i%2Cj)%20%5Cin%20R%7D%20%5Cell_%7BP%7D(i%2Cj)%20-%20%5Cmathcal%7BL%7D(P)%20%5C%5C%0A%26%3D%20%5Csum_%7B(i%2Cj)%20%5Cin%20R%7D%20%5Cell_%7BP'%7D(i%2Cj)%20-%20%5Csum_%7B(i%2Cj)%20%5Cin%20R%7D%20%5Cell_P(i%2Cj)%20%5C%5C%0A%26%3D%20%5Csum_%7B(i%2Cj)%20%5Cin%20R%7D%20%5C%7C%20P'(i%2Cj)%20-%20I(i%2Cj)%20%5C%7C%5E2%20-%20%5Csum_%7B(i%2Cj)%20%5Cin%20R%7D%20%5Cell_%7BP%7D(i%2Cj)%20%5C%5C%0A%26%3D%20%5Csum_%7B(i%2Cj)%20%5Cin%20R%7D%20%5C%7C%20b_%7B%5Cvec%20C%2CO%7D(P(i%2Cj))%20-%20I(i%2Cj)%20%5C%7C%5E2%20-%20%5Csum_%7B(i%2Cj)%20%5Cin%20R%7D%20%5C%7C%20P(i%2Cj)%20-%20I(i%2Cj)%20%5C%7C%5E2.%5Cend%7Baligned%7D" alt="\begin{aligned}
\Delta \mathcal{L}_P(B) := \mathcal{L}(T_{B}(P)) - \mathcal{L}(P) &amp;= \sum_{i,j} \ell_{P'}(i,j) - \mathcal{L}(P) \\
&amp;= \sum_{(i,j) \in R} \ell_{P'}(i,j) + \sum_{(i,j) \notin R} \ell_{P'}(i,j) - \mathcal{L}(P) \\
&amp;= \sum_{(i,j) \in R} \ell_{P'}(i,j) + \sum_{(i,j) \notin R} \| P'(i,j) - I(i,j) \|^2 - \mathcal{L}(P) \\
&amp;= \sum_{(i,j) \in R} \ell_{P'}(i,j) + \sum_{(i,j) \notin R} \| P(i,j) - I(i,j) \|^2 - \mathcal{L}(P) \\
&amp;= \sum_{(i,j) \in R} \ell_{P'}(i,j) + \sum_{(i,j) \notin R} \ell_{P}(i,j) - \mathcal{L}(P) \\
&amp;= \sum_{(i,j) \in R} \ell_{P'}(i,j) + \sum_{i,j} \ell_{P}(i,j) - \sum_{(i,j) \in R} \ell_{P}(i,j) - \mathcal{L}(P) \\
&amp;= \sum_{(i,j) \in R} \ell_{P'}(i,j) + \mathcal{L}(P) - \sum_{(i,j) \in R} \ell_{P}(i,j) - \mathcal{L}(P) \\
&amp;= \sum_{(i,j) \in R} \ell_{P'}(i,j) - \sum_{(i,j) \in R} \ell_P(i,j) \\
&amp;= \sum_{(i,j) \in R} \| P'(i,j) - I(i,j) \|^2 - \sum_{(i,j) \in R} \ell_{P}(i,j) \\
&amp;= \sum_{(i,j) \in R} \| b_{\vec C,O}(P(i,j)) - I(i,j) \|^2 - \sum_{(i,j) \in R} \| P(i,j) - I(i,j) \|^2.\end{aligned}" />
This formula avoids the need to sum over the whole image, instead just
summing over the region painted on.

Suppose we have a painting <img src="https://i.upmath.me/svg/P" alt="P" />, and we have a set of brushes
<img src="https://i.upmath.me/svg/S%20%5Csubseteq%20%5Cmathcal%7BR%7D%20%5Ctimes%20%5Cmathcal%20C%20%5Ctimes%20%5Cmathcal%20O" alt="S \subseteq \mathcal{R} \times \mathcal C \times \mathcal O" /> to choose
from. We want to choose the brush <img src="https://i.upmath.me/svg/B%20%5Cin%20S" alt="B \in S" /> that minimizes
<img src="https://i.upmath.me/svg/%5Cmathcal%7BL%7D(T_%7BB%7D(P))" alt="\mathcal{L}(T_{B}(P))" />. This is equivalent to minimizing
<img src="https://i.upmath.me/svg/%5CDelta%20%5Cmathcal%7BL%7D_P(B)" alt="\Delta \mathcal{L}_P(B)" />, since <img src="https://i.upmath.me/svg/%5CDelta%20%5Cmathcal%7BL%7D_P(B)" alt="\Delta \mathcal{L}_P(B)" /> and
<img src="https://i.upmath.me/svg/%5Cmathcal%7BL%7D(T_%7BB%7D(P))" alt="\mathcal{L}(T_{B}(P))" /> differ by the <img src="https://i.upmath.me/svg/%5Cmathcal%7BL%7D(P)" alt="\mathcal{L}(P)" />, which is
constant for all brushes. Thus when finding brushes we will minimize
<img src="https://i.upmath.me/svg/%5CDelta%20%5Cmathcal%7BL%7D_P" alt="\Delta \mathcal{L}_P" />. 

Choosing a brush region
=======================

Now that we are able to pick the optimal brush color and opacity given a
brush region, we are concerned with picking the best brush region. There
are way too many regions to check by brute force, and you can't
differentiate with respect to the region to try to do gradient based
optimization, so what we will use random-restart hill climbing.

To generate random neighbors, "parameterized brush regions" will be
used. We have a number of parameterized brush region classes, labelled 1
through <img src="https://i.upmath.me/svg/K" alt="K" />. Each parameterized brush region class <img src="https://i.upmath.me/svg/k%20%5Cin%20%5C%7B1%2C%5Cdots%2CK%5C%7D" alt="k \in \{1,\dots,K\}" />
has associated with it a set of possible parameters <img src="https://i.upmath.me/svg/%5CTheta_k" alt="\Theta_k" /> (usually
<img src="https://i.upmath.me/svg/%5CTheta_k%20%5Csubseteq%20%5Cmathbb%7BR%7D%5Ep" alt="\Theta_k \subseteq \mathbb{R}^p" /> for some <img src="https://i.upmath.me/svg/p%20%5Cin%20%5Cmathbb%7BN%7D" alt="p \in \mathbb{N}" />), and a
brush in that class is identified by its parameters. The class also has
a random variable <img src="https://i.upmath.me/svg/G_k%20%3A%20%5COmega_k%20%5Cto%20%5CTheta_k" alt="G_k : \Omega_k \to \Theta_k" /> which generates random
parameters and is to be used by random search and random-restart hill
climbing; a function <img src="https://i.upmath.me/svg/r_k%20%3A%20%5CTheta_k%20%5Cto%20%5Cmathcal%7BR%7D" alt="r_k : \Theta_k \to \mathcal{R}" /> which returns the
region of a brush given its parameters; and a function
<img src="https://i.upmath.me/svg/n_k%20%3A%20%5CTheta_k%20%5Cto%20%5CTheta_k" alt="n_k : \Theta_k \to \Theta_k" /> which generates a "neighbor" of a
parameterized region and is to be used by hill climbing.

Let's make a function that given a region and a painting, gives the
optimal color and opacity:
<img src="https://i.upmath.me/svg/p%20%3A%20%5Cmathcal%20R%20%5Ctimes%20%5Cmathcal%20I%20%5Cto%20%5Cmathcal%20C%20%5Ctimes%20%5Cmathcal%20O" alt="p : \mathcal R \times \mathcal I \to \mathcal C \times \mathcal O" /> and
<img src="https://i.upmath.me/svg/p(R%2C%20P)%20%3D%20(%5Cvec%20C%5E*%2C%20O%5E*)" alt="p(R, P) = (\vec C^*, O^*)" />.

Algorithm
=========

The basic algorithm is as follows:

1.  Start with <img src="https://i.upmath.me/svg/P_0" alt="P_0" /> having all ones (white).

2.  For iteration <img src="https://i.upmath.me/svg/t%3D0" alt="t=0" /> to max iter:

    1.  Set
        <img src="https://i.upmath.me/svg/L_%7B%5Ctext%7Bbest%7D%7D%20%3D%200%2C%20(k_%7B%5Ctext%7Bbest%7D%7D%2C%20%5Ctheta_%7B%5Ctext%7Bbest%7D%7D)%20%3D%20None" alt="L_{\text{best}} = 0, (k_{\text{best}}, \theta_{\text{best}}) = None" />

    2.  For each brush class <img src="https://i.upmath.me/svg/k%3D1" alt="k=1" /> through <img src="https://i.upmath.me/svg/K" alt="K" />:

    3.  1.  <img src="https://i.upmath.me/svg/(%5Ctheta%2C%20L)%20%3A%3D%20%5Ctextsc%7BHillClimbing%7D(%5Clambda%20%5Ctheta%20.%20%5CDelta%20%5Cmathcal%7BL%7D_%7BP_k%7D(r_k(%5Ctheta)%2C%20p(r_k(%5Ctheta)%2C%20P_t))%2C%20G_k%2C%20n_k)" alt="(\theta, L) := \textsc{HillClimbing}(\lambda \theta . \Delta \mathcal{L}_{P_k}(r_k(\theta), p(r_k(\theta), P_t)), G_k, n_k)" />

        2.  If <img src="https://i.upmath.me/svg/L%20%3C%20L_%7B%5Ctext%7Bbest%7D%7D" alt="L &lt; L_{\text{best}}" />, then

            -   <img src="https://i.upmath.me/svg/(k_%5Ctext%7Bbest%7D%2C%5Ctheta_%5Ctext%7Bbest%7D)%20%3A%3D%20(k%2C%20%5Ctheta)" alt="(k_\text{best},\theta_\text{best}) := (k, \theta)" />

            -   <img src="https://i.upmath.me/svg/L_%5Ctext%7Bbest%7D%20%3A%3D%20L" alt="L_\text{best} := L" />

    4.  <img src="https://i.upmath.me/svg/P_%7Bt%2B1%7D%20%3A%3D%20T_%7Br_%7Bk_%5Ctext%7Bbest%7D%7D(%5Ctheta_%5Ctext%7Bbest%7D)%7D(P_t)" alt="P_{t+1} := T_{r_{k_\text{best}}(\theta_\text{best})}(P_t)" />
