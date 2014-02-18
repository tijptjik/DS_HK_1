## Lesson 6

* [Assumptions of Linear Regressions](http://pareonline.net/getvn.asp?n=2&v=8), [Breaking them](http://www.basic.northwestern.edu/statguidefiles/linreg_ass_viol.html), [Transformations](http://www.basic.northwestern.edu/statguidefiles/linreg_alts.html#Transformations) & [Discussion](http://andrewgelman.com/2013/08/04/19470/) 
 	* linearity
 	* reliability of measurement
 	* homoscedasticity
 	* normality

or 

 	1. Validity.
 	1. Additivity and linearity.
 	1. Independence of errors
 	1. Equal variance of errors
 	1. Normality of errors

[Polynominals & MultiColinearity](https://www.stat.fi/isi99/proceedings/arkisto/varasto/kim_0574.pdf)

[Minimizing the Effects of Colinearity](ftp://ftp.bgu.ac.il/shacham/publ_papers/IandEC_36_4405_97.pdf)

[Adding Polynominals](http://stats.stackexchange.com/questions/58739/polynomial-regression-using-scikit-learn)

[Multilinearit Graphs](https://onlinecourses.science.psu.edu/stat501/node/85)

Alternatively, or formally, we can check the tolerance values or variance inflation 
ratio (VIF) to investigate possible collinearity.

We define the tolerance as 1/ R2

and VIF 1/Tolerance

The less the tolerance’s value, (or it is closer to zero, or < 0.1), the worse of the 
collinearity. This is conforms to the formula: as tolerance close to zero then R2
is closer to 1, meaning a stronger linear relation.

It is not surprising, since VIF is the reciprocal of the tolerance, then the larger of the 
value of VIF, the worse the collinearity!

Usually, if VIF is greater than 10 we should consider it a warning sign! Under the situation when there is collinearity, we may reasonably consider using only one of the correlated variables (ignore the other one, it does not matter which one to choose staying in the model). 


[Norms](http://rorasa.wordpress.com/2012/05/13/l0-norm-l1-norm-l2-norm-l-infinity-norm/)

[Polynomial interpolation](http://scikit-learn.org/stable/auto_examples/linear_model/plot_polynomial_interpolation.html)
[SciKit polyfit](http://docs.scipy.org/doc/numpy/reference/generated/numpy.polyfit.html)
[OverFitting](http://www.astroml.org/sklearn_tutorial/general_concepts.html#linearly-separable-data)
[Using Polynomials](http://www.astroml.org/sklearn_tutorial/practical.html)
[Common Mistakes on Intepreting Regressions](https://www.ma.utexas.edu/users/mks/statmistakes/regressioncoeffs.html)
[Orthoganl Polinominals](http://dlmf.nist.gov/18.4)
[L-1 and L-2 Norms](http://rorasa.wordpress.com/2012/05/13/l0-norm-l1-norm-l2-norm-l-infinity-norm/)
[Effects of L1](http://cseweb.ucsd.edu/~saul/teaching/cse291s07/L1norm.pdf)

Extra
[Introduction to Regression](http://dss.princeton.edu/online_help/analysis/regression_intro.htm)
[Good Regression Overview](http://www.stat.purdue.edu/~jennings/stat514/stat512notes/topic3.pdf)
[Introduction to Multivariate Regression](http://www.apec.umn.edu/grad/jdiaz/IntroductiontoRegression.pdf)
[OSL in Matrix Form](http://www.stanford.edu/~mrosenfe/soc_meth_proj3/matrix_OLS_NYU_notes.pdf)
[Regression with Gradient Descent Function](https://github.com/KartikTalwar/OnlineCourses/blob/master/Stanford%20University/Machine%20Learning/02.%20Linear%20Regression%20with%20One%20Variable.md#cost-function)
[Multicollinearity and Singularity](http://dss.wikidot.com/multicollinearity-and-singularity)
[Multivariate Linear Regression Models](http://www.public.iastate.edu/~maitra/stat501/lectures/MultivariateRegression.pdf)
[Orthogonal Polynomials](https://www.cs.iastate.edu/~cs577/handouts/orthogonal-polys.pdf)
[Orthogonal Functions & Expansions](http://web.hep.uiuc.edu/home/serrede/P435/Lecture_Notes/P435_Supp_HO_01.pdf)
[The Analytic Theory of Matrix Orthogonal Polynomials](http://www.emis.de/journals/SAT/papers/11/11.pdf)
[Transfomations](http://www.biostat.jhsph.edu/~iruczins/teaching/jf/ch8.pdf)
[Regularization with Ridge penalties, the Lasso, and the Elastic Net for Regression with Optimal Scaling Transformations](https://openaccess.leidenuniv.nl/bitstream/handle/1887/12096/04.pdf?sequence=18)
[Vector And Matrix Norms](http://www-solar.mcs.st-andrews.ac.uk/~clare/Lectures/num-analysis/Numan_chap1.pdf)
[Introduction to Multicolinearity](https://onlinecourses.science.psu.edu/stat501/node/79)
[Multicolinearity](http://www.chsbs.cmich.edu/fattah/courses/empirical/multicollinearity.html)
[Testing the assumptions of linear regression](http://people.duke.edu/~rnau/testing.htm)
[Multicolinearity](http://en.wikipedia.org/wiki/Multicollinearity)

[Examples]
[Fitting sine wave data by polynomials](http://nbviewer.ipython.org/github/carljv/Will_it_Python/blob/master/MLFH/ch6/ch6.ipynb)

```python
import numpy as np

np.random.seed(42)
x = np.random.random(20)
y = np.sin(2 * x)

# fit a 1st-degree polynomial (i.e. a line) to the data
p = np.polyfit(x, y, 1) # [ 0.97896174  0.20367395]
x_new = np.random.random(3)
y_new = np.polyval(p, x_new)  # evaluate the polynomial at x_new
print abs(np.sin(x_new) - y_new) # [ 0.22826933  0.20119119  0.20166572]
```

```python
%pylab inline

import numpy as np
np.random.seed(42)
x = np.random.random(20)
y = np.sin(2 * x)
p = np.polyfit(x, y, 1)  # fit a 1st-degree polynomial (i.e. a line) to the data
print p  # slope and intercept

x_new = np.random.random(3)
y_new = np.polyval(p, x_new)  # evaluate the polynomial at x_new
print abs(np.sin(x_new) - y_new)

import pylab as plt
def plot_fit(x, y, p):
    xfit = np.linspace(0, 1, 1000)
    yfit = np.polyval(p, xfit)
    plt.scatter(x, y, c='b')
    plt.plot(xfit, yfit)
    plt.xlabel('x')
    plt.ylabel('y')
    
plot_fit(x, y, p)

```

1. Increase the number of training points N. This might give us a training set with more coverage, and lead to greater accuracy.
2. Increase the degree d of the polynomial. This might allow us to more closely fit the training data, and lead to a better result
3. Add more features. If we were to, for example, perform a linear regression using x, x√, x−1, or other functions, we might hit on a functional form which can better be mapped to the value of y.


```python
def test_func(x, err=0.5):
    return np.random.normal(10 - 1. / (x + 0.1), err)

def compute_error(x, y, p):
    yfit = np.polyval(p, x)
    return np.sqrt(np.mean((y - yfit) ** 2))
```

[Cost Minimization Problem w/ Lagrangian](http://www.youtube.com/watch?v=PlZ0Mgu-9RY)


### Signs of Multicolinearity

* A regression coefficient is not significant even though, theoretically, that variable should be highly correlated with Y.
* When you add or delete an X variable, the regression coefficients change dramatically.
* You see a negative regression coefficient when your response should increase along with X.
* You see a positive regression coefficient when the response should decrease as X increases.
* Your X variables have high pairwise correlations.

