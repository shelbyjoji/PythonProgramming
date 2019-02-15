
# Part II
# --------------------
# easiest simulation
# ---------------------
# Binomial
# ---------------------
import pandas as pd
import numpy as np

# procedure 1
n1=10
p1 = 0.5
s1 = np.random.binomial(n1, p1, 10)
s1
np.mean(s1)
np.var(s1)

# 100 repetation
s2 = np.random.binomial(n1, p1, (100,10))
s2
np.mean(s2)
np.var(np.concatenate(s2))

# ---------------------
# Poisson
# ---------------------
# procedure 1

lamda = 3.5
s1 = np.random.poisson(lamda,n1)
s1
np.mean(s1)
np.var(s1)

# 100 repetition
s2 = np.random.poisson(lamda,(100,n1))
s2
np.mean(s2)
np.var(s2)

# -----------------
# Uniform
# -----------------
s1 = np.random.uniform(0,20,n1).round(2)
s1
np.mean(s1)
np.var(s1)

# 100 repetition
s2 = np.random.uniform(0,20,(100,n1)).round(2)
np.mean(s2)
np.var(s2)

# --------------------
# Normal
# --------------------
mu = 10
sd1= 2
s1 = np.random.normal(mu,sd1,n1)
s1
np.mean(s1)
np.var(s1)
# 100 repetition
s2 = np.random.normal(mu,sd1,(100,n1)).round(2)
np.mean(s1)
np.var(s2)

# -------------------
# Exponential
# -------------------
lamda= 2
s1 = np.random.exponential(lamda,n1).round(2)
s1
np.mean(s1)
np.var(s1)
# 100 repetition
s2 = np.random.exponential(lamda,(100,n1)).round(2)
s2
np.mean(s2)
np.var(s2)

# ----------------------
# TIME VARIANT
# -----------------------

def simdata(n):
    j = [i for i in range(1,n+1)]
    y00 = np.random.uniform(1,100,size=n).round()
    y11 = np.random.geometric(p=.3,size=n)
    y12 = np.random.binomial(n=10,p=.5,size=n)
    y13 = np.random.poisson(lam=4,size=n).round(4)
    y14 = np.random.exponential(scale=7,size=n).round(2)
    y15 = np.random.normal(10,2,size=n).round(2)
    m1 = pd.DataFrame({'sample_size':j,'geom':y11,'binom':y12,
                       'poisson':y13, 'expon':y14,'normal':y15,'age':y00})
    ds = [rows for _, rows in m1.groupby('age')]
    return(ds)
simdata(1000)


def foo(n):
    X = simdata(n)
    res = pd.concat([param_estimation(Xi) for Xi in X],axis=1)
    return res

def param_estimation(Xi):
    m = 1/(Xi.geom.mean() + 1)
    m1 = 1 if m > 1 else m
    m2 = Xi.binom.values.mean()/10
    m3  = Xi.poisson.mean()
    m4 = Xi.expon.mean()
    m5 = Xi.normal.mean()
    distribution_list = ['geom','binom','poisson','expon','normal']
    sim_mean = pd.Series([m1,m2,m3,m4,m5],index=distribution_list)
    sim_mean.name = Xi.age.values[0] #get the age
    return sim_mean
replications = 100
[foo(2500) for rep in range(replications)]


n = 2500
B = 100
