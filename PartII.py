
# Part II
# --------------------
# SIMULATION MADE EASY
# --------------------

import pandas as pd
import numpy as np

# ---------------------
# Binomial distribution
# ---------------------
# Question 01
# Take a random sample from Binomial distribution with p=0.7, size=10 and number of trials is 10.
# This means that you have 10 coins, which you toss 10 times and probability of appearing head is .70.

n1 = 10   # Number of trials
p1 = 0.5   # Probablity of success
s1 = np.random.binomial(n1, p1, 10)
print(s1)

# Compute the mean and variance of this simulated data. True mean & variance are ‘np’ & ‘np(1-p)’.
np.mean(s1)
np.var(s1)

# Repeat this procedure for 100 times. This repetition must be random.
# Then estimate the mean and variance. Compare it with true mean and variance of binomial distribution.

# Next we do 100 repetition of the procedure. Number of trials, probability and size remains the same
s2 = np.random.binomial(n1, p1, (100, 10))
print(s2)

np.mean(s2)
np.var(s2)

# ---------------------
# Poisson distribution
# ---------------------
# If events are occurring randomly and independently, Then X, the number of events in a fixed unit of time, has
# poisson distribution. P(X=x) = (λ^x * exp(-λ)) / x!. Note x can take on any value to infinity.
# Mean of poisson distribution is λ, and so is variance
# Binomial distribution tend towards poisson distribution  as n goes to infinity, p goes to zero, and np stays constant
# The poisson distribution with λ = np closely approximates binomial distribution  if n is large and p is small.
# The Poisson distribution is the limit of the binomial distribution for large N.

# Take a random sample from Poisson distribution with λ = 3 and number of observations is 10.
# λ is an unknown parameter from the Poisson distribution, which must be provided for data simulation.
# Compute the mean and variance of this simulated data. As I said before true mean & variance is λ.

lam = 3.5  # set the λ value
n1 = 10
s1 = np.random.poisson(lam, n1)  # n1 is the size
print(n1)
np.mean(s1)
np.var(s1)

# Repeat this procedure for 100 times. This repetition must be random.
# Then estimate the mean and variance. Compare it with true mean and variance of Poisson distribution.
# 100 repetition
s2 = np.random.poisson(lam, (100, n1))
print(s2)
np.mean(s2)
np.var(s2)

# -----------------
# Uniform distribution
# -----------------

# A uniform distribution, sometimes also known as a rectangular distribution,
# is a distribution that has constant probability. A uniform probability distribution is the one that corresponds to
# the intuitive idea of all values (of the random variable) being "equally likely".
# In the case of a one dimensional discrete random variable with finitely many values, this is exactly what it means.
# If X is a random variable that takes values x1,x2,…,xn, it follows a uniform distribution
# if P[X=xi] = c, a fixed constant, for i=1,2,…,n. Obviously, c=1/n
# The mean of the distribution is  x¯ = ∑ (xi) /n, and the variance is ∑ (xi − x¯ )2 / n.

# Example: The outcomes of the roll of a fair die form a uniform distribution. The values are 1,2,3,4,5,6,
# each with probability 1/6. On the other hand, the sum of the two outcomes on rolling
# two fair dice does not follow a uniform distribution.


# Take a random sample of size 10 from the Uniform distribution with a=0 (low value) and b=20 (high value).
# Compute the mean and variance of this simulated data. True mean is (a+b)/2 and true variance is 〖(b-a)〗^2 / 12.
# Mathematical Form: f(x) =  1/(b-a); a<x<b.


# np.random.uniform(low, high, size)
s1 = np.random.uniform(0, 20, n1).round(2)
print(s1)

np.mean(s1)
np.var(s1)

# Repeat this procedure for 100 times. This repetition must be random.
# Then estimate mean and variance. Compare it with the true mean and variance of the uniform distribution.
# 100 repetition

s2 = np.random.uniform(0, 20, (100, n1)).round(2)
print(s2)
np.mean(s2)
np.var(s2)

# --------------------
# Normal distribution
# --------------------

# The normal distributions occurs often in nature. For example, it describes the commonly occurring distribution
# of samples influenced by a large number of tiny, random disturbances, each with its own unique distribution [2].
# The probability density function of the normal distribution is often called the bell curve
# because of its characteristic shape

# Take a random sample of size 10 from the normal distribution with mean 10 and standard deviation 3.
# Compute the mean and variance of this simulated data. True mean and variance are 10 and 3.

mu = 10  # mean
sd1 = 2  # variance
s1 = np.random.normal(mu, sd1, n1)  # mean, variance, size
print(s1)
np.mean(s1)
np.var(s1)

# Repeat this procedure for 100 times. This repetition must be random.
# Then estimate mean and variance and compare it with true mean and variance.
# 100 repetition
s2 = np.random.normal(mu, sd1, (100, n1)).round(2)
np.mean(s1)
np.var(s2)

# The probability density for the Gaussian distribution is:
# f(x)= (1/(σ√(2π))) e^( -1/2  * 〖((x-μ)/σ)〗^2 );      -∞<x<∞.

# Difference between normal and uniform distribution:
# 1. Normal has infinite support, uniform has finite support
# 2. Normal has a single most likely value, uniform has every allowable value equally likely
# 3. Uniform has a piecewise constant density, normal has a continuous bell shaped density
# 4. Normal distributions arise from the central limit theorem, uniforms do not

# -------------------
# Exponential distribution
# -------------------

# The exponential distribution is the probability distribution that describes the time between events
# in a Poisson point process, i.e., a process in which events occur continuously and independently
# at a constant average rate. The exponential distribution is a continuous analogue of the geometric distribution.
# It describes many common situations, such as the size of raindrops measured over many rainstorms,
# or the time between page requests to Wikipedia


# Take a random sample of size 10 from the Exponential distribution with λ = 5. Compute the mean and variance
# of this simulated data. True mean and variance are 5 and 25.

lam = 5  # lambda value
n1 = 10  # size
s1 = np.random.exponential(lam, n1).round(2)
print(s1)
np.mean(s1)
np.var(s1)

# Repeat this procedure for 100 times. This repetition must be random. Then estimate mean and variance.
# Compare it with true mean and variance. Now we do 100 repetition
s2 = np.random.exponential(lam, (100, n1)).round(2)
print(s2)
np.mean(s2)
np.var(s2)

# Mathematical form: f(x)= 1/λ e^( (-1/λ) * x);0<x<∞.

# ----------------------
# TIME VARIANT -  FIXED PARAMETER
# -----------------------

# Time-invariant simulation from parametric probability models:
# In this simulation design, we will simulate data from various probability models under the assumption
# that data will be generated with fixed parameter value at each time point.

# The probability models are
# 1. binomial (parameter, n=10, p=0.6),
# 2. Poisson (parameter, λ=4)
# 3. Geometric (0.3),
# 4. Exponential (λ=1/7),
# 5. normal (μ=10, σ=2).

def simdata(n):
    j = [i for i in range(1, n+1)]
    y00 = np.random.uniform(1, 100, size=n).round()
    y11 = np.random.geometric(p=.3, size=n)
    y12 = np.random.binomial(n=10, p=.5, size=n)
    y13 = np.random.poisson(lam=4, size=n).round(4)
    y14 = np.random.exponential(scale=7, size=n).round(2)
    y15 = np.random.normal(10, 2, size=n).round(2)
    m1 = pd.DataFrame({'sample_size': j,
                       'geom': y11,
                       'binom': y12,
                       'poisson': y13,
                       'expon': y14,
                       'normal': y15,
                       'Uniform': y00})

    ds = [rows for _, rows in m1.groupby('Uniform')]
    return ds

simulate = simdata(1000)
print(simulate)



def param_estimation(Xi):
    m = 1/(Xi.geom.mean() + 1)
    m1 = 1 if m > 1 else m
    m2 = Xi.binom.values.mean()/10
    m3 = Xi.poisson.mean()
    m4 = Xi.expon.mean()
    m5 = Xi.normal.mean()
    distribution_list = ['geom', 'binom','poisson', 'expon', 'normal']
    sim_mean = pd.Series([m1, m2, m3, m4, m5], index=distribution_list)
    sim_mean.name = Xi.Uniform.values[0] #get the uniform
    return sim_mean

def foo(n):
    X = simdata(n)
    res = pd.concat([param_estimation(Xi) for Xi in X],axis=1)
    return res

replications = 100
[foo(2500) for rep in range(replications)]


# -------------------------------------
# TIME VARIANT -  Variable Parameter
# -------------------------------------

# In this simulation design, we will simulate data from various probability models under the assumption
# that data will be generated with varying parameter values at each time point. The probability models
# are binomial (parameter, n=10, p lies between 0.4 and 0.7), Poisson (parameter λ lies .65 and10),
# Geometric (parameter ϴ lies between 0.2 and 0.3), Exponential (parameter λ lies between .3 and .7),
# and normal (μ lies between 10 and 20, σ=2).

def simdata(n) :
    assert n % 25 == 0, "n must be a multiple of 25"
    unif = np.random.uniform
    binom = np.random.binomial
    j = np.arange(1, 26).tolist()
    reps = int(n / 25)  # python doesn't automatically repeat numpy arrays and lists, so you have to manually
    # make it do so:         4* [1,2,3,4] for a LIST makes    [1,2,3,4,1,2,3,4,1,2,3,4]
    y11 = np.random.geometric(p=unif(0.2, 0.8, size=25).tolist() * reps, size=n)
    y12 = np.random.binomial(n=10, p=unif(0.4, 0.7, size=25).tolist() * reps, size=n)
    y13 = np.random.poisson(lam=(binom(p=0.65, n=10, size=25)).tolist() * reps, size=n)
    y14 = np.random.exponential(scale=unif(1 / .7, 1 / .3, size=25).tolist() * reps, size=n).round(2)
    y15 = np.random.normal(unif(10, 20, size=25).tolist() * reps, 2, size=n).round(2)

    m1 = pd.DataFrame({'sample_size' : n, 'geom' : y11, 'binom' : y12,
                       'poisson' : y13, 'expon' : y14, 'normal' : y15, 'age' : j * reps})
    ds = [rows for _, rows in m1.groupby('age')]
    return (ds)

def param_estimation(Xi) :
    m = 1 / (Xi.geom.mean() + 1)
    m1 = 1 if m > 1 else m
    m2 = Xi.binom.values.mean() / 10
    m3 = Xi.poisson.mean()
    m4 = Xi.expon.mean()
    m5 = Xi.normal.mean()
    distribution_list = ['geom', 'binom', 'poisson', 'expon', 'normal']
    sim_mean = pd.Series([m1, m2, m3, m4, m5], index=distribution_list)
    sim_mean.name = Xi.age.values[0]  # get the age
    return sim_mean

def foo(n) :
    X = simdata(n)
    res = pd.concat([param_estimation(Xi) for Xi in X], axis=1)
    return res

replications = 100
[foo(2500) for rep in range(replications)]
