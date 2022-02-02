*If you are reading this README.md on GitHub, I encourage you to read the file `paint_regression_description.pdf` instead because the typesetting is better there (especially if you're using dark mode GitHub in which case the math is not readable here!)*

Introduction
============

*Example of the results. The source image is from https://thispersondoesnotexist.com/*

https://user-images.githubusercontent.com/33670185/151718369-7b9d419a-b148-4234-9fe0-44c612b0e147.mp4


The goal of my paint regression project is to try to get the computer to
"paint" an image, by using a highly simplified model of how an artist
paints a picture: this is how one would opt to paint if they wanted to
get their painting as close as possible to a reference image with the
least amount of effort. The goal of the program is to create a painting
that is as close as possible to the source image after a given fixed
number of brush strokes. A "brush stroke" is a region of pixels in the
image, together with a color and an opacity, and applying a brush stroke
is coloring in the painting with the color and opacity in the region.

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
of a painting:

<img src="https://i.upmath.me/svg/%5Cbegin%7Baligned%7D%0A%5Cmathcal%7BL%7D(P)%20%26%3A%3D%20%5Csum_%7Bi%3D1%7D%5Em%20%5Csum_%7Bj%3D1%7D%5En%20%5Csum_%7Bk%3D1%7D%5Ec%20(%20P(i%2Cj%2Ck)%20-%20I(i%2Cj%2Ck)%20)%5E2%20%5C%5C%0A%26%3D%20%5Csum_%7Bi%3D1%7D%5Em%20%5Csum_%7Bj%3D1%7D%5En%20%5Cunderbrace%7B%5C%7C%20P(i%2Cj)%20-%20I(i%2Cj)%20%5C%7C%5E2%7D_%7B%5Cell_P(i%2Cj)%7D%5Cend%7Baligned%7D" alt="\begin{aligned}
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
\end{cases}" />

where

<img src="https://i.upmath.me/svg/b_%7B%5Cvec%20C%2CO%7D(%5Cvec%20x)%20%3D%20O%20%5Cvec%20C%20%2B%20(1-O)%20%5Cvec%20x." alt="b_{\vec C,O}(\vec x) = O \vec C + (1-O) \vec x." />

Now I will write change in the loss from applying a brush <img src="https://i.upmath.me/svg/B%3D(R%2C%5Cvec%20C%2CO)" alt="B=(R,\vec C,O)" />. I will call this change of loss

<img src="https://i.upmath.me/svg/%5CDelta%20%5Cmathcal%7BL%7D_P(B)%20%3A%3D%20%5Cmathcal%7BL%7D(T_%7BB%7D(P))%20-%20%5Cmathcal%7BL%7D(P)." alt="\Delta \mathcal{L}_P(B) := \mathcal{L}(T_{B}(P)) - \mathcal{L}(P)." />

Write <img src="https://i.upmath.me/svg/P'%20%3D%20T_%7BB%7D(P)" alt="P' = T_{B}(P)" /> as shorthand. First, note that

<img src="https://i.upmath.me/svg/%5Csum_%7B(i%2Cj)%20%5Cnotin%20R%7D%20%5Cell_%7BP'%7D(i%2Cj)%20%3D%20%5Csum_%7B(i%2Cj)%20%5Cnotin%20R%7D%20%5C%7C%20P'(i%2Cj)%20-%20I(i%2Cj)%20%5C%7C%5E2%0A%3D%20%5Csum_%7B(i%2Cj)%20%5Cnotin%20R%7D%20%5C%7C%20P(i%2Cj)%20-%20I(i%2Cj)%20%5C%7C%5E2%0A%3D%20%5Csum_%7B(i%2Cj)%20%5Cnotin%20R%7D%20%5Cell_P(i%2Cj)." alt="\sum_{(i,j) \notin R} \ell_{P'}(i,j) = \sum_{(i,j) \notin R} \| P'(i,j) - I(i,j) \|^2
= \sum_{(i,j) \notin R} \| P(i,j) - I(i,j) \|^2
= \sum_{(i,j) \notin R} \ell_P(i,j)." />

<img src="https://i.upmath.me/svg/%5Cbegin%7Baligned%7D%0A%5CDelta%20%5Cmathcal%7BL%7D_P(B)%20%3D%20%5Cmathcal%7BL%7D(T_%7BB%7D(P))%20-%20%5Cmathcal%7BL%7D(P)%20%26%3D%20%5Csum_%7Bi%2Cj%7D%20%5Cell_%7BP'%7D(i%2Cj)%20-%20%5Csum_%7Bi%2Cj%7D%20%5Cell_P(i%2Cj)%20%5C%5C%0A%26%3D%20%5Csum_%7B(i%2Cj)%20%5Cin%20R%7D%20%5Cell_%7BP'%7D(i%2Cj)%20%2B%20%5Csum_%7B(i%2Cj)%20%5Cnotin%20R%7D%20%5Cell_%7BP'%7D(i%2Cj)%20-%20%5Csum_%7B(i%2Cj)%20%5Cin%20R%7D%20%5Cell_P(i%2Cj)%20-%20%5Csum_%7B(i%2Cj)%20%5Cnotin%20R%7D%5Cell_P(i%2Cj)%20%5C%5C%0A%26%3D%20%5Csum_%7B(i%2Cj)%20%5Cin%20R%7D%20%5Cell_%7BP'%7D(i%2Cj)%20%2B%20%5Csum_%7B(i%2Cj)%20%5Cnotin%20R%7D%20%5Cell_%7BP%20%7D(i%2Cj)%20-%20%5Csum_%7B(i%2Cj)%20%5Cin%20R%7D%20%5Cell_P(i%2Cj)%20-%20%5Csum_%7B(i%2Cj)%20%5Cnotin%20R%7D%5Cell_P(i%2Cj)%20%5C%5C%0A%26%3D%20%5Csum_%7B(i%2Cj)%20%5Cin%20R%7D%20%5Cell_%7BP'%7D(i%2Cj)%20-%20%5Csum_%7B(i%2Cj)%20%5Cin%20R%7D%20%5Cell_P(i%2Cj)%20%5C%5C%0A%26%3D%20%5Csum_%7B(i%2Cj)%20%5Cin%20R%7D%20%5C%7C%20P'(i%2Cj)%20-%20I(i%2Cj)%20%5C%7C%5E2%20-%20%5Csum_%7B(i%2Cj)%20%5Cin%20R%7D%20%5Cell_%7BP%7D(i%2Cj)%20%5C%5C%0A%26%3D%20%5Csum_%7B(i%2Cj)%20%5Cin%20R%7D%20%5C%7C%20b_%7B%5Cvec%20C%2CO%7D(P(i%2Cj))%20-%20I(i%2Cj)%20%5C%7C%5E2%20-%20%5Csum_%7B(i%2Cj)%20%5Cin%20R%7D%20%5C%7C%20P(i%2Cj)%20-%20I(i%2Cj)%20%5C%7C%5E2.%5Cend%7Baligned%7D" alt="\begin{aligned}
\Delta \mathcal{L}_P(B) = \mathcal{L}(T_{B}(P)) - \mathcal{L}(P) &amp;= \sum_{i,j} \ell_{P'}(i,j) - \sum_{i,j} \ell_P(i,j) \\
&amp;= \sum_{(i,j) \in R} \ell_{P'}(i,j) + \sum_{(i,j) \notin R} \ell_{P'}(i,j) - \sum_{(i,j) \in R} \ell_P(i,j) - \sum_{(i,j) \notin R}\ell_P(i,j) \\
&amp;= \sum_{(i,j) \in R} \ell_{P'}(i,j) + \sum_{(i,j) \notin R} \ell_{P }(i,j) - \sum_{(i,j) \in R} \ell_P(i,j) - \sum_{(i,j) \notin R}\ell_P(i,j) \\
&amp;= \sum_{(i,j) \in R} \ell_{P'}(i,j) - \sum_{(i,j) \in R} \ell_P(i,j) \\
&amp;= \sum_{(i,j) \in R} \| P'(i,j) - I(i,j) \|^2 - \sum_{(i,j) \in R} \ell_{P}(i,j) \\
&amp;= \sum_{(i,j) \in R} \| b_{\vec C,O}(P(i,j)) - I(i,j) \|^2 - \sum_{(i,j) \in R} \| P(i,j) - I(i,j) \|^2.\end{aligned}" />
This formula avoids the need to sum over the whole image, instead just
summing over the region painted on. Note that the function
<img src="https://i.upmath.me/svg/%5CDelta%20%5Cmathcal%7BL%7D_P" alt="\Delta \mathcal{L}_P" /> always outputs a non-positive value.

Suppose we have a painting <img src="https://i.upmath.me/svg/P" alt="P" />, and we have a set of brushes
<img src="https://i.upmath.me/svg/S%20%5Csubseteq%20%5Cmathcal%7BR%7D%20%5Ctimes%20%5Cmathcal%20C%20%5Ctimes%20%5Cmathcal%20O" alt="S \subseteq \mathcal{R} \times \mathcal C \times \mathcal O" /> to choose
from. We want to choose the brush <img src="https://i.upmath.me/svg/B%20%5Cin%20S" alt="B \in S" /> that minimizes
<img src="https://i.upmath.me/svg/%5Cmathcal%7BL%7D(T_%7BB%7D(P))" alt="\mathcal{L}(T_{B}(P))" />. This is equivalent to minimizing
<img src="https://i.upmath.me/svg/%5CDelta%20%5Cmathcal%7BL%7D_P(B)" alt="\Delta \mathcal{L}_P(B)" />, since <img src="https://i.upmath.me/svg/%5CDelta%20%5Cmathcal%7BL%7D_P(B)" alt="\Delta \mathcal{L}_P(B)" /> and
<img src="https://i.upmath.me/svg/%5Cmathcal%7BL%7D(T_%7BB%7D(P))" alt="\mathcal{L}(T_{B}(P))" /> differ by <img src="https://i.upmath.me/svg/%5Cmathcal%7BL%7D(P)" alt="\mathcal{L}(P)" />, which is constant
for all brushes. Thus when finding brushes we will minimize
<img src="https://i.upmath.me/svg/%5CDelta%20%5Cmathcal%7BL%7D_P" alt="\Delta \mathcal{L}_P" />.

Choosing color and opacity given a brush region
===============================================

Suppose that we have the current painting <img src="https://i.upmath.me/svg/P" alt="P" /> to paint on, and we
already have a chosen region to paint on, <img src="https://i.upmath.me/svg/R" alt="R" />. Then we can choose the
color <img src="https://i.upmath.me/svg/%5Cvec%20C" alt="\vec C" /> and opacity <img src="https://i.upmath.me/svg/O" alt="O" /> that minimize
<img src="https://i.upmath.me/svg/%5CDelta%20%5Cmathcal%7BL%7D_P(R%2C%5Cvec%20C%2CO)." alt="\Delta \mathcal{L}_P(R,\vec C,O)." /> As shown above,
<img src="https://i.upmath.me/svg/%5CDelta%20%5Cmathcal%7BL%7D_P(B)%20%3D%20%5Csum_%7B(i%2Cj)%20%5Cin%20R%7D%20%5C%7C%20b_%7B%5Cvec%20C%2CO%7D(P(i%2Cj))%20-%20I(i%2Cj)%20%5C%7C%5E2%20-%20%5Csum_%7B(i%2Cj)%20%5Cin%20R%7D%20%5C%7C%20P(i%2Cj)%20-%20I(i%2Cj)%20%5C%7C%5E2" alt="\Delta \mathcal{L}_P(B) = \sum_{(i,j) \in R} \| b_{\vec C,O}(P(i,j)) - I(i,j) \|^2 - \sum_{(i,j) \in R} \| P(i,j) - I(i,j) \|^2" />
and the second term is constant with respect to <img src="https://i.upmath.me/svg/%5Cvec%20C" alt="\vec C" /> and <img src="https://i.upmath.me/svg/O" alt="O" />, so we
want to minimize the following quantity:

<img src="https://i.upmath.me/svg/%20%5Csum_%7B(i%2Cj)%20%5Cin%20R%7D%20%5C%7C%20b_%7B%5Cvec%20C%2CO%7D(P(i%2Cj))%20-%20I(i%2Cj)%20%5C%7C%5E2" alt=" \sum_{(i,j) \in R} \| b_{\vec C,O}(P(i,j)) - I(i,j) \|^2" />

And if we
write <img src="https://i.upmath.me/svg/%5Cvec%20%5Calpha%20%3D%20O%5Cvec%20C" alt="\vec \alpha = O\vec C" /> and <img src="https://i.upmath.me/svg/%5Cbeta%20%3D%201-O" alt="\beta = 1-O" />, then

<img src="https://i.upmath.me/svg/b_%7B%5Cvec%20C%2CO%7D(%5Cvec%20x)%20%3D%20O%20%5Cvec%20C%20%2B%20(1-O)%20%5Cvec%20x%20%3D%20%5Cvec%20%5Calpha%20%2B%20%5Cbeta%20%5Cvec%20x" alt="b_{\vec C,O}(\vec x) = O \vec C + (1-O) \vec x = \vec \alpha + \beta \vec x" />

and it doesn't matter if we optimize <img src="https://i.upmath.me/svg/%5Cvec%20%5Calpha" alt="\vec \alpha" /> and <img src="https://i.upmath.me/svg/%5Cbeta" alt="\beta" /> instead
of <img src="https://i.upmath.me/svg/%5Cvec%20C" alt="\vec C" /> and <img src="https://i.upmath.me/svg/O" alt="O" /> because you can go back and forth between the two
forms. If we write <img src="https://i.upmath.me/svg/R%20%3D%20%5C%7B%20(i_1%2Cj_1)%2C%20%5Cdots%2C%20(i_N%2Cj_N)%20%5C%7D%2C" alt="R = \{ (i_1,j_1), \dots, (i_N,j_N) \}," />

and then write

<img src="https://i.upmath.me/svg/X%20%3A%3D%20%5Cbegin%7Bbmatrix%7DP(i_1%2Cj_1)%20%5C%5C%20%5Cvdots%20%5C%5C%20P(i_N%2Cj_N)%20%5Cend%7Bbmatrix%7D%2C%20%5Cquad%20Y%20%3A%3D%20%5Cbegin%7Bbmatrix%7D%20I(i_1%2Cj_1)%20%5C%5C%20%5Cvdots%20%5C%5C%20I(i_N%2Cj_N)%20%5Cend%7Bbmatrix%7D%2C" alt="X := \begin{bmatrix}P(i_1,j_1) \\ \vdots \\ P(i_N,j_N) \end{bmatrix}, \quad Y := \begin{bmatrix} I(i_1,j_1) \\ \vdots \\ I(i_N,j_N) \end{bmatrix}," />

then we can rewrite:

<img src="https://i.upmath.me/svg/%5Csum_%7B(i%2Cj)%20%5Cin%20R%7D%20%5C%7C%20b_%7B%5Cvec%20C%2CO%7D(P(i%2Cj))%20-%20I(i%2Cj)%20%5C%7C%5E2%20%3D%0A%5Csum_%7Bi%3D1%7D%5EN%20%5C%7C%20%5Cvec%20%5Calpha%20%2B%20%5Cbeta%20X_i%20-%20Y_i%20%5C%7C%5E2%20%3D%20%5Csum_%7Bi%3D1%7D%5EN%20%5Csum_%7Bj%3D1%7D%5Ec%20%5B%5Cvec%20%5Calpha_j%20%2B%20%5Cbeta%20X_%7Bi%2Cj%7D%20-%20Y_%7Bi%2Cj%7D%20%5D%5E2." alt="\sum_{(i,j) \in R} \| b_{\vec C,O}(P(i,j)) - I(i,j) \|^2 =
\sum_{i=1}^N \| \vec \alpha + \beta X_i - Y_i \|^2 = \sum_{i=1}^N \sum_{j=1}^c [\vec \alpha_j + \beta X_{i,j} - Y_{i,j} ]^2." />

We want to choose <img src="https://i.upmath.me/svg/%5Cvec%20%5Calpha" alt="\vec \alpha" /> and <img src="https://i.upmath.me/svg/%5Cbeta" alt="\beta" /> that minimize this quantity.
By taking partial derivatives, one can obtain the following solution
<img src="https://i.upmath.me/svg/(%5Cvec%20%5Calpha%5E*%2C%5Cbeta%5E*)" alt="(\vec \alpha^*,\beta^*)" />.

Write the mean of the <img src="https://i.upmath.me/svg/k" alt="k" /> component of <img src="https://i.upmath.me/svg/X" alt="X" /> and <img src="https://i.upmath.me/svg/Y" alt="Y" /> as

<img src="https://i.upmath.me/svg/%5Coverline%7BX_%7B%3A%2Ck%7D%7D%20%3A%3D%20%5Cfrac%7B1%7D%7BN%7D%20%5Csum_%7Bi%3D1%7D%5EN%20X_%7Bi%2Ck%7D%2C%20%5Cquad%0A%5Coverline%7BY_%7B%3A%2Ck%7D%7D%20%3A%3D%20%5Cfrac%7B1%7D%7BN%7D%20%5Csum_%7Bi%3D1%7D%5EN%20Y_%7Bi%2Ck%7D%2C" alt="\overline{X_{:,k}} := \frac{1}{N} \sum_{i=1}^N X_{i,k}, \quad
\overline{Y_{:,k}} := \frac{1}{N} \sum_{i=1}^N Y_{i,k}," />

and <img src="https://i.upmath.me/svg/S_%7Bxx%7D" alt="S_{xx}" /> and <img src="https://i.upmath.me/svg/S_%7Bxy%7D" alt="S_{xy}" /> are

<img src="https://i.upmath.me/svg/S_%7Bxx%7D%20%3A%3D%20%5Csum_%7Bi%3D1%7D%5EN%20%5Csum_%7Bj%3D1%7D%5Ec%20(X_%7Bi%2Cj%7D%20-%20%5Coverline%7BX_%7B%3A%2Cj%7D%7D)%5E2%2C%20%5Cquad%0AS_%7Bxy%7D%20%3A%3D%20%5Csum_%7Bi%3D1%7D%5EN%20%5Csum_%7Bj%3D1%7D%5Ec%20(X_%7Bi%2Cj%7D%20-%20%5Coverline%7BX_%7B%3A%2Cj%7D%7D)%20(Y_%7Bi%2Cj%7D%20-%20%5Coverline%7BY_%7B%3A%2Cj%7D%7D)" alt="S_{xx} := \sum_{i=1}^N \sum_{j=1}^c (X_{i,j} - \overline{X_{:,j}})^2, \quad
S_{xy} := \sum_{i=1}^N \sum_{j=1}^c (X_{i,j} - \overline{X_{:,j}}) (Y_{i,j} - \overline{Y_{:,j}})" />

then the solution is:

<img src="https://i.upmath.me/svg/%5Calpha%5E*_k%20%3D%20%5Coverline%7BY_%7B%3A%2Ck%7D%7D%20-%20%5Cbeta%5E*%20%5Coverline%7BX_%7B%3A%2Ck%7D%7D%2C" alt="\alpha^*_k = \overline{Y_{:,k}} - \beta^* \overline{X_{:,k}}," />

and

<img src="https://i.upmath.me/svg/%5Cbeta%5E*%20%3D%20%5Cfrac%7BS_%7Bxy%7D%7D%7BS_%7Bxx%7D%7D%20%5Cquad%20%5Ctext%7Bif%20%7D%20S_%7Bxx%7D%20%5Cneq%200." alt="\beta^* = \frac{S_{xy}}{S_{xx}} \quad \text{if } S_{xx} \neq 0." />

If <img src="https://i.upmath.me/svg/S_%7Bxx%7D%20%3D%200" alt="S_{xx} = 0" />, then it must be the case that <img src="https://i.upmath.me/svg/S_%7Bxy%7D%20%3D%200" alt="S_{xy} = 0" /> in which case
any value of <img src="https://i.upmath.me/svg/%5Cbeta%5E*" alt="\beta^*" /> works (so if <img src="https://i.upmath.me/svg/S_%7Bxx%7D%20%3D%200" alt="S_{xx} = 0" /> but <img src="https://i.upmath.me/svg/S_%7Bxy%7D%20%5Cneq%200" alt="S_{xy} \neq 0" />,
then there was an arithmetic error).

Once we obtain <img src="https://i.upmath.me/svg/(%5Cvec%20%5Calpha%5E*%2C%20%5Cbeta%5E*)" alt="(\vec \alpha^*, \beta^*)" /> which are optimal, we can set
<img src="https://i.upmath.me/svg/O%5E*%20%3D%201%20-%20%5Cbeta%5E*" alt="O^* = 1 - \beta^*" />, <img src="https://i.upmath.me/svg/%5Cvec%20C%5E*%20%3D%20%5Cfrac%7B%5Cvec%20%5Calpha%5E*%7D%7BO%5E*%7D" alt="\vec C^* = \frac{\vec \alpha^*}{O^*}" />. If
<img src="https://i.upmath.me/svg/O%5E*%20%5Cnotin%20%5Cmathcal%20O%20%3D%20%5B0%2C1%5D" alt="O^* \notin \mathcal O = [0,1]" /> or
<img src="https://i.upmath.me/svg/%5Cvec%20C%5E*%20%5Cnotin%20%5Cmathcal%20C%20%3D%20%5B0%2C1%5D%5Ec" alt="\vec C^* \notin \mathcal C = [0,1]^c" />, then we could try to do a
constrained optimization, but I'm lazy so in this case let's just reject
the region <img src="https://i.upmath.me/svg/R" alt="R" /> and pick another region.

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

The basic algorithm is as follows (note that in step 2(c)i there is an
anonymous function using the lambda calculus notation):

1.  Start with <img src="https://i.upmath.me/svg/P_0" alt="P_0" /> having all ones (white).

2.  For iteration <img src="https://i.upmath.me/svg/t%3D0" alt="t=0" /> to max iter:

    1.  Set
        <img src="https://i.upmath.me/svg/L_%7B%5Ctext%7Bbest%7D%7D%20%3D%200%2C%20(k_%7B%5Ctext%7Bbest%7D%7D%2C%20%5Ctheta_%7B%5Ctext%7Bbest%7D%7D)%20%3D%20None" alt="L_{\text{best}} = 0, (k_{\text{best}}, \theta_{\text{best}}) = None" />

    2.  For each brush class <img src="https://i.upmath.me/svg/k%3D1" alt="k=1" /> through <img src="https://i.upmath.me/svg/K" alt="K" />:

    3.  1.  <img src="https://i.upmath.me/svg/(%5Ctheta%2C%20L)%20%3A%3D%20%5Ctextsc%7BHillClimbing%7D(%5Clambda%20%5Ctheta%20.%20%5CDelta%20%5Cmathcal%7BL%7D_%7BP_t%7D(r_k(%5Ctheta)%2C%20p(r_k(%5Ctheta)%2C%20P_t))%2C%20G_k%2C%20n_k)" alt="(\theta, L) := \textsc{HillClimbing}(\lambda \theta . \Delta \mathcal{L}_{P_t}(r_k(\theta), p(r_k(\theta), P_t)), G_k, n_k)" />

        2.  If <img src="https://i.upmath.me/svg/L%20%3C%20L_%7B%5Ctext%7Bbest%7D%7D" alt="L &lt; L_{\text{best}}" />, then

            -   <img src="https://i.upmath.me/svg/(k_%5Ctext%7Bbest%7D%2C%5Ctheta_%5Ctext%7Bbest%7D)%20%3A%3D%20(k%2C%20%5Ctheta)" alt="(k_\text{best},\theta_\text{best}) := (k, \theta)" />

            -   <img src="https://i.upmath.me/svg/L_%5Ctext%7Bbest%7D%20%3A%3D%20L" alt="L_\text{best} := L" />

    4.  <img src="https://i.upmath.me/svg/P_%7Bt%2B1%7D%20%3A%3D%20T_%7Br_%7Bk_%5Ctext%7Bbest%7D%7D(%5Ctheta_%5Ctext%7Bbest%7D)%2Cp(r_%7Bk_%5Ctext%7Bbest%7D%7D(%5Ctheta_%5Ctext%7Bbest%7D)%2CP_t)%7D(P_t)" alt="P_{t+1} := T_{r_{k_\text{best}}(\theta_\text{best}),p(r_{k_\text{best}}(\theta_\text{best}),P_t)}(P_t)" />
