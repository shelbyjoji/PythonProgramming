import os
import sympy as sym
import pandas as pd
import numpy as np
import seaborn as sns
import scipy
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.stats as ss
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from statsmodels.stats.proportion import proportions_ztest
from matplotlib.pyplot import pie, axis, show
from scipy.integrate import quad
from sympy import *
from sympy.solvers.solveset import linsolve
from scipy import stats
from pprint import pprint
import scipy.special

os.getcwd()  # find my working directory
pd.options.mode.chained_assignment = None  # disable copy warning

# reading csv data
beaver1 = pd.read_csv('beaver1.csv')
beaver2 = pd.read_csv('beaver2.csv')
trees = pd.read_csv('trees.csv')
USArrests = pd.read_csv('USArrests.csv')
warpbreaks = pd.read_csv('warpbreaks.csv')
iris = pd.read_csv('iris.csv')
mtcars = pd.read_csv('mtcars.csv')

# Q1 - to check number of observations and variables - beaver1 data
print(beaver1.shape)

# Q2 -  check the types of variables in iris data
print(iris.dtypes)

# Q3  - select 100 rows from beaver1 and beaver2 and denote by b1 and b2
b1 = beaver1.iloc[:100, :]
b2 = beaver2.iloc[:100, :]

# Q4 - create ID variables with labels 1:100 fro b1 and b2
b1['ID'] = b1.index.values + 1
b2['ID'] = b2.index.values + 1

# Q5 -  Merge b1 and b2 dataset by ID variable  so that merged dataset has one ID variable
# Other variables stay side by side
merged = b1.merge(b2, left_on='ID', right_on='ID')

# Q6 - Recreate ID variable for b2 data. Labels take values from 101:200
b2['ID'] = b2.index.values + 101

# Q7 -  Merge b1 and b2 by ID variable such that b2 stays below b1
append = b1.append(b2)

# Q8 -  create categorical variable time.cat from beaver1
# The bins will be as follows: [0,935),[935,`1450], (1450, infinity)
beaver1['time.cat'] = pd.cut(beaver1['time'].values, bins=[0, 935, 1451, np.inf], right=False, labels=[1, 2, 3],
                             include_lowest=False)

# Q9 - Find mean for each individual variable corresponding to each label of species
iris.groupby('Species').mean()

# Q10 - Find frequency table fro species variable in iris data
pd.crosstab(index=iris["Species"], columns="count")

# Q11 - Find the cross tabulation frequencies for wool and tension variables of warpbreaks data
pd.crosstab(index=warpbreaks["wool"], columns=warpbreaks['tension'])

# Q12 - Find box plot of sepal length for each category of species in iris data

sns.set(style="whitegrid")
ax = sns.boxplot(x="Species", y="Sepal.Length", data=iris)

# Q13 -  for USArrests create Murder.Percent with percent value
# expressing murder as percent of assault

USArrests['Murder.Percent'] = round((USArrests["Murder"] / USArrests["Assault"]) * 100, 2)
USArrests['Murder.Percent'] = USArrests['Murder.Percent'].astype(str) + '%'

# Q14 -  Delete percent sign from Murder.Percent variable
USArrests['Murder.Percent'] = USArrests['Murder.Percent'].str.replace('%', '')
USArrests['Murder.Percent'] = USArrests['Murder.Percent'].astype(float)

# Q15 - Remove rownames from USArrests data -  basically dropped state
USArrests = USArrests.drop(['State'], axis=1)

# Q16 - Subset data by selecting variables Murder, Assault and Rape
Variables = ["Murder", "Assault", "Rape"]
U1 = USArrests[Variables]

# Q17 - Create Total Crime by adding Murder, Assault and Rape
U1["Total.Crime"] = U1["Murder"] + U1["Assault"] + U1["Rape"]

# Q18 - Create histogram of all three variables
plt.subplot(221)
plt.hist(U1["Murder"])
plt.title('Histogram of Murder')

plt.subplot(222)
plt.hist(U1["Assault"])
plt.title('Histogram of Assault')

plt.subplot(223)
plt.hist(U1["Rape"])
plt.title('Histogram of Rape')

plt.subplot(224)
plt.hist(U1["Total.Crime"])
plt.title('Histogram of Total.Crime')

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.35)
plt.show()

# Q19 - Sort by Murder Variable
USArrests.sort_values(["Murder"], axis=0, ascending=True, inplace=True)

# Q20 - Construct a bar chart for tension variable - warpbreaks data
sns.set(style="whitegrid")
sns.countplot(x="tension", data=warpbreaks).set_title("Bar chart for tension variable")

# Q21 - Construct a pie chart for tension variable - warpbreaks data
sums = warpbreaks.tension.groupby(warpbreaks.tension).count()
axis('equal')
pie(sums, labels=sums.index)
plt.title("Pie chart for tension variable")
show()

# Q22 -  Find area of 10 circles with radius from 1:10
print("Radius Area")


def area_list() :
    for r in range(1, 10) :
        print(r, "  \t", round((np.pi) * r ** 2, 2))


area_list()


# Q23 - Sum of 1:10000 for 1/exp(x)


def f(x) :
    y = 1 / (np.exp(x))
    return y


i = np.linspace(1, 10000, 10000)
print("Answer: ", sum(f(i)))

# Q24 Compute sum of 1 to 30, sum of 1 to 10 (i^4) / (3+j)

i = np.linspace(1, 30, 30).astype(int)
j = np.linspace(1, 10, 10).astype(int)
print(sum([i ** 4 / (3 + j) for i in range(1, 30) for j in range(1, 10)]))


# Q25 -  integration

def integrant(x) :
    y = (x ** 15) * np.exp(-7 * x)
    return y


ans, err = quad(integrant, 0, np.inf)
print(ans)


# Q26 -  integration


def integrant(x) :
    y = (x ** 15) * ((1 - x) ** 30)
    return y


ans, err = quad(integrant, 0, 1)
print(ans)

# Q27 - compute

x = np.linspace(1, 5, 5)
y = np.linspace(2, 6, 5)
z = np.linspace(3, 7, 5)


def f(x, y, z) :
    z1 = z ** 2
    result = ((np.exp(x)) - np.log(z1)) / (5 + y)
    print(result)


print(f(x, y, z))

# Q28 Solve equation

solution = sym.solve('x**2 - (33*x)+1', 'x')
solution

# Q29 Solve the system
x, y, z = sym.symbols(('x', 'y', 'z'))
eq_system = sym.Matrix([[7, 10, 4, 90], [12, 45, 34, 100], [23, 23, 123, 300]])
solution = linsolve(eq_system, (x, y, z))
x, y, z = next(iter(solution))
print("Solution:")
print("x =", x)
print("y =", y)
print("z =", z)

# Q30 -  Inverse of matrix

A = np.array([[25, 30, 37], [23, 27, 33], [21, 27, 43]])
print("Inverse of A:")
print(np.linalg.inv(A))
print()

# Q31 - Matrix operations

U = np.array([[15, 10, 17], [13, 17, 13], [11, 17, 13]])
V = np.array([[55, 50, 57], [53, 57, 53], [51, 57, 53]])
A = U + V
B = U.dot(V)
print("Matrix Addition \n U + V:")
print(A)
print()

print("Matrix Dot product Multiplication \n A * B:")
print(B)
print()


# Q32 - Rank of Matrix

print("Matrix Rank of U+V:")
print(np.linalg.matrix_rank(A))

print("Matrix Rank of UV:")
print(np.linalg.matrix_rank(B))

# Q33 -  Chi2 test of independence
pd.crosstab(index=warpbreaks["wool"], columns=warpbreaks['tension'])
aa = np.array([[9, 9, 9], [9, 9, 9]])
print(stats.chi2_contingency(aa))

# chi-squared value and p value
# returns Chi2, pvalue, degrees of freedom, expected as ndarray

# Q34 - Test whether means for variables mpg of mtcars data is 25 or not. Left tailed is manually calculated
stats.ttest_1samp(mtcars["mpg"], 25)

# Q35 - Test on equality of two mean - mpg corresponding to two categories of vs of mtcars data
stats.ttest_ind(mtcars[mtcars.vs == 0].mpg, mtcars[mtcars.vs == 1].mpg)

# Q36 - proportion test for setosa of Species variable in iris data

count = len(iris.loc[iris['Species'] == 'setosa'])
nobs = len(iris['Species'].dropna())
alpha_value = .05
print("Count ", count)
print("Number of Observation ", nobs)
stat, pval = proportions_ztest(count, nobs, alpha_value)
print("Test statistic: ", stat,"\np-value : ", pval)


# Q37 - Normality test on Petal.Length on iris data   - Shapiro test
stats.shapiro(iris["Petal.Length"]) # test statistic and p-value as output

# Q38 - Log transformation on Petal.Length on iris data and do Shapiro test
x = np.log(iris["Petal.Length"])
stats.shapiro(x)

# Q39 -  BoxCox transformation and check normality
xt, _ = stats.boxcox(iris["Petal.Length"])
stats.shapiro(xt)

# Visualize the difference
fig = plt.figure()
ax1 = fig.add_subplot(211)
prob = stats.probplot(x, dist=stats.norm, plot=ax1)
ax1.set_xlabel('')
ax1.set_title('Prob plot against normal distribution') # Log transformed

ax2 = fig.add_subplot(212)
prob = stats.probplot(xt, dist=stats.norm, plot=ax2)
ax2.set_title('Prob plot after BoxCox transformation')

# Q40 - Test whether median temp of beaver1 and median temp of beaver2 are equal or not.
u1, p_value1 = stats.mannwhitneyu(beaver1.temp, beaver2.temp)
print("Test statistic: ",u1,"\np-value: ", p_value1)

# Q41 -  Single loop on iris data

datalist = [] # note that this is a list and not a dataframe

# subset the data by getting columns you need from original data d2
d9 = iris[["Species","Petal.Length"]]
ds = [rows for _, rows in d9.groupby('Species')]

# I just wanted to see my list nicely to analyze the list. You could just use ds, and it will do it.

pprint(ds)

len(ds)

ds[0]

for i in range(len(ds)):
       datalist.append(pd.DataFrame({
      'Mean.Petal.Length': [np.mean(ds[i]["Petal.Length"])],
      'Median.Petal.Length': [np.median(ds[i]["Petal.Length"])],
      'Minimum.Petal.Length': [min(ds[i]["Petal.Length"])],
      'Maximum.Petal.Length': [max(ds[i]["Petal.Length"])],
      'Standard_dev.Petal.Length': [np.std(ds[i]["Petal.Length"])]}))

print("Please Ignore the zeroes")
pd.concat(datalist)
pprint(datalist)

# Q42 -  Correlation matrix

variables = ['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']
df = iris[variables]
corr = df.corr()
sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)

# Q43 - Binomial distribution
n1 = 20   # Number of trials
p1 = 0.7   # Probablity of success
s1 = np.random.binomial(n1, p1, 20)
print(s1)
# Compute the mean and variance of this simulated data. True mean & variance are ‘np’ & ‘np(1-p)’.
np.mean(s1)
np.var(s1)

# Repeat this procedure for 100 times. This repetition must be random.
# Then estimate the mean and variance. Compare it with true mean and variance of binomial distribution.
# Next we do 100 repetition of the procedure. Number of trials, probability and size remains the same
s2 = np.random.binomial(n1, p1, (100, 20))
print(s2)

np.mean(s2)
np.var(s2)

# Q44 - Draw the graph
scipy.special.factorial(6)

def f(x):
    f1 = scipy.special.factorial(x)
    y = np.exp(x)/f1
    return y

x = np.linspace(2, 10, 9)
plt.plot(x, f(x))
plt.show()


# Q45 - step function plot
x = np.linspace(1, 10, 100) # make 1000 evenly spaced values from -2.99 to 2.99

def tmpFn(xi):
    if xi < 0:
        return 2*(xi ** 2) + (5 * xi) + 3

    elif xi >= 0 and xi < 2:
        return (9*xi) + 3

    elif xi >= 2:
        return 7*(xi**2) + (5*xi) -17

tmpFn = np.vectorize(tmpFn) # when you input a list or array, the new function will apply the function to each element within it

# similar to how a for loop works
fx = tmpFn(x)
plt.style.use("seaborn")
plt.plot(x, fx)
exclusive_x_coords = [x[0],x[[-1]]]
exclusive_y_coords = [fx[0],fx[[-1]]]
plt.scatter(exclusive_x_coords,exclusive_y_coords,color='w',edgecolors='blue',zorder=5) # marking the exclusive points on
# the graph.
# z order = 5 makes the points opaque
plt.xlabel("x",fontweight=15)
plt.ylabel("f(x)",fontsize=15)
plt.title("Line Plot of x as a Continuous Function of f(x)",fontsize=15,fontweight='bold')
plt.show()



# Q46 -  regression

df = trees[["Height","Volume"]].dropna()
df["Intercept"] = 1
X_train, X_test, y_train, y_test = train_test_split(df[["Intercept",'Height']].values, df['Volume'].values, test_size=0.3, random_state=42)


rm= sm.OLS(y_train,X_train,intercept=True).fit()
intercept,wt_coef = rm.params
print(rm.summary())
print("intercept: ",intercept,"\nParameters: ", wt_coef )


# Q47 - find predicted values from independent values
predictions=rm.fittedvalues
print("Predicted values from train set: ", predictions)

# Q48 - Predicted value from test dataset
new_predictions = rm.predict(X_test)
print("Predicted values from test set: ", new_predictions)
print("True labels from test set: ", y_test)

# Q49 -  Minimum, maximum and quartiles of mtcars
mtcars.describe()

# Q50  -  Find missing values of columns in mtcars data
mtcars.isnull().sum(axis=0)