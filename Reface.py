
# Part I

import pandas as pd
import numpy as np

#Q1
# reading csv data
d1 = pd.read_csv('C:/Users/HOMEPCUSER/Desktop/Spring 19/PythonProgramming/PROJECTDATA.csv')
print(d1)

# reading text data
text = pd.read_table('C:/Users/HOMEPCUSER/Desktop/Spring 19/PythonProgramming/ProjectData.txt')
print(text)

#Q2
# to get the dimension of dataset
d1.shape
np.shape(d1)

#Q3
# count variables  that are integer and numeric
d1.info()
d1.dtypes

# report number of variables
print(d1.columns)

#Q4
# Delete the ID variable from the data set
del d1['ID']
del d1['Unnamed: 0']

#get the head
d1.head(10)

# get the tail
d1.tail(10)

#Q5
#missing values for each variables -  it reports the null value
d1.isnull().sum()

#Q6
#obtain summary statistics -  in python describe is for summary statistics
d1.describe()

#reports summary statistic for specific variable
d1.describe()[['SBP', 'DBP']]

#subset the data and then describe
d3 = d1[['SBP', 'DBP']]
d3.describe()

#Q7
#create column for missing values in each row - axis = 0 for column, axis = 1 for rows
d1["MISSING"] = d1.isnull().sum(axis=1)

#Q8 & #Q9 & #Q10
#Here we drop rows that has missing value.
#Create a new data d2 by deleting any observation (any row) that has missing value
d2 = d1.dropna()
d2.shape # reporting number of rows in new data
d2.describe() # reporting new summary statistic

#let us check for missing values in d2
d2.isnull().sum() # none right?


#Q11
#Use the apply function (R code) to report standard deviation for numerical variables only
#This requires finding all numerical variables

#Look at the actual dataframe and find all categorical variables. You cannot use dtypes to find out categorical
#We will remove all categorical variable
irrelevant_variables_list = ["RACE","INCOME","SMOKE","BREAST","WAIST","EVENT"]
#next line gets the names of all variable that are numerical. All variables - categorical gives numerical.
relevant_columns = [variable for variable in d2.columns if variable not in irrelevant_variables_list]
relevant_columns
#Now you get standard deviation of numerical variable only
d2[relevant_columns].std()
d2[relevant_columns].mean()


#Q12
#create ABP (Average Blood Pressure) variable by averaging SBP and DBP for each row of th
initial_bp_variables = ['SBP', 'DBP']
#Remember axis =1 gives you rows
#mean(axis=1) gives average of selected rows
d2["ABP"] = d2[initial_bp_variables].mean(axis=1)
d2.tail()


d4 = d2[["RACE","SBP"]]
d4.tail()
d4

# subsetting data
all_quant_bp_variables = ['SBP','DBP','ABP']
d2[all_quant_bp_variables].head()
d2.head()

#Q13
#Creating categorical variable from numerical
#First, you need to create an upper limit
abp_limit = d2['ABP'].max()+1
abp_limit

# Because we will be binning the data using exclusive right binning functions,
# We make sure that the highest value is still included in the new categorical
# Label when creating bins
bpc_ranges = [0,85,100,abp_limit] # The bins will be as follows: [0,85),[85,`100), [100, infinity)
# the labels of the new categprical variable
bpc_labels = [3,2,1]
# This function creates categorical variables out of quantitative
d2['BPC'] = pd.cut(d2['ABP'].values, bins = bpc_ranges, right = False, labels = bpc_labels, include_lowest = False )
d2.head()

#now we create a display of all new variables we dealt with
# variables by "binning" them.
new_bp_variables = ['ABP','BPC']
# with lists, we can join them together using list1 + list2
new_bp_variables
initial_bp_variables
all_bp_variables = new_bp_variables+initial_bp_variables
print("Preview of All Blood Pressure Variables:")
print(d2[all_bp_variables])


# Q14
# frequency table for the BPC variable -  frequency table for a categorical variable
# Here we count the categorical variable
pd.crosstab(index=d2["BPC"],columns="Count")
pd.crosstab(index=d2["SMOKE"],columns="Count")

# Q15
# Frequncy table for BPC stratified by race
# Categorical variable by categorical variable
pd.crosstab(index=d2["BPC"],columns=d2["RACE"])

#ABP is for average BP. It does not make any sense to use ABP for frequncy table as it is quantitative

# Note: Use Groupby to split category and get mean, var, std and so on

# Q17
# Look smoke is categorical
# find variance of TC for SMOKE variable
# Getting results by grouping
bysmoke = d2.groupby('SMOKE')
bysmoke.mean()
bysmoke.std()#to get variance
K = bysmoke.var()#to get variance
K[["TC"]]

# Q16
#FInd mean of TC for RACE variable
bysmoke = d2.groupby('RACE')
k= bysmoke.mean()
bysmoke.std()
k[["TC"]]


# Q18
# In this section, we draw bar chart and pie chart
import matplotlib.pyplot as plt # import plotting library

# Specify colors and label
cols = ['r','g'] #assign colors
labels = ['NonSmoker','Smoker'] #assign labels


# table command for frequency table -  category by count
pd.crosstab(index=d2["SMOKE"], columns="count")

sizes = [74, 1868] # get this from above freqency table  -  we take count of category
explode = [0, 0.1] # this will make the piece of pie protrude out; None will keep it within the circle

# for pie chart
plt.pie(sizes,explode = explode, labels = labels, colors=cols)
# for axis and titles
plt.title("Pie Chart of Race Variable")
plt.axis('equal')

# bar chart
x = ['NonSmoker','Smoker'] # name categories
y = sizes

# X is category and y is count
plt.bar(x,y)
plt.title("Bar Chart of Smoke Variable")
plt.xlabel("Smoking Status")
plt.ylabel("Frequency")

# you could do the same for RACE and INCOME

# Q19
# Category by category on count
# Frequncy table for RACE stratified by INCOME
pd.crosstab(index=d2["RACE"], columns=d2["INCOME"])

#Frequncy table for INCOME stratified by RACE
pd.crosstab(index=d2["INCOME"], columns=d2["RACE"])


# stacked bar chart -  One above other -  stacked

import matplotlib.pyplot as plt

# Frequncy table for RACE stratified by INCOME
pd.crosstab(index=d2["RACE"], columns=d2["INCOME"])

# From the above frequency table you get A (for RACE 1) and B(for RACE 2)
A = [14,18,13,80,149,166,167,249,113]
B = [87,71,57,188,164,143,86,155,22]
X = range(9) # get the range of income
X
plt.bar(X, A, color='b',label="Race1")
plt.bar(X, B, color='r', bottom = A, label="Race2")

plt.title("Stack Bar Chart Income and Race")
plt.xlabel("Income Status")
plt.ylabel("Frequency")
plt.legend(bbox_to_anchor=(.22,.98), loc=1) # Place legend box

#Q20

#Now replot the same as unstacked barplot

# set width of bar
barWidth = 0.25

# Set position of bar on X axis
r1 = np.arange(len(A))
r1

r2 = [x + barWidth for x in r1]
r2

# Make the plot
plt.bar(r1, A, color='orange', width=barWidth, edgecolor='white', label='Race1')
plt.bar(r2, B, color='steelblue', width=barWidth, edgecolor='white', label='Race2')

# Add xticks on the middle of the group bars. Without xticks you would not get bar labels under each bar
plt.xlabel('Income Status', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(A))], [1,2,3,4,5,6,7,8,9]) # for bar labels

# Create legend & Show graphic
plt.legend()# you could decide where to place this by looking at previous plotting in previous question
plt.title("Grouped Bar Chart for Income and Race")
plt.ylabel("Frequency")
plt.show()




#------------
# Q21
#------------
# dot plot for DBP5.
x = d2["DBP"] # get all the values of DBP - Quantitative variable
x
plt.plot(x,'bo') # bo is for blue circles. You could use go, for green, or maybe something else
plt.title('Dotplot.DBP')

# subset data by category
#Here you table DBP by race. Basically you are extracting it into two dataset x1 and x2
x1 = d2[(d2.RACE == 1)]["DBP"]
x2 = d2[(d2.RACE == 2)]["DBP"]
x1 # look how x1 is

# plotting two seperate graphs
plt.plot(x1,'bo')
plt.title('Dotplot CC')
plt.grid(True)

plt.plot(x2,'bo')
plt.title('Dotplot.AA')
plt.grid(True)

plt.show()

# ------------
# Q22
# ------------

# Draw histogram for SBP  - quantitative variable
y = d2["SBP"]
plt.hist(y)
plt.title("Histogram of SBP")
plt.xlabel("Values of SBP")
plt.ylabel("Frequency")

# Draw two separate histograms for SBP when RACE=1 and RAce=2
# two histogram together in one graph of quantitative variable
y1 = d2[(d2.RACE == 1)]["SBP"]
y2 = d2[(d2.RACE == 2)]["SBP"]
plt.hist(y1)
plt.title('SBP for african american')
plt.xlabel("Values of SBP")
plt.hist(y2)
plt.title('SBP for caucasian')
plt.xlabel("Values of SBP")

# ------------
# Q23
# ------------
#Draw boxplot for SBP - a quantitative variable
plt.boxplot(y)
plt.title("Boxplot of SBP")



# Draw two separate boxplot for SBP when RACE=1 and RAce=2
# Boxplot of quantitative variable by categories
# two box plot together. If you run it together, it will be in same graph.
plt.boxplot(y1)
plt.title('Box Plot of SBP for african american')

plt.boxplot(y2)
plt.title('Box Plot of SBP for caucasian')

# ----------------------
# Q24, all 9 graphs
# ----------------------

# Creating subplots
# here we are subplotting on 3 by 3 grid
# 331,332,333
# 334, 335, 336
# 337, 338,339

# 1. three dot plot

plt.subplot(331)
plt.plot(x,'bo')
plt.title('Dot plot of DBP')

plt.subplot(332)
plt.plot(x1,'bo')
plt.title('Dot plot of CC')
plt.grid(True)

plt.subplot(333)
plt.plot(x2,'bo')
plt.title('Dot plot of AA')
plt.grid(True)

# 2. Three histogram
plt.subplot(334)
plt.hist(y)
plt.title("Histogram of SBP")
plt.xlabel("Values of SBP")
plt.ylabel("Frequency")

plt.subplot(335)
plt.hist(y1)
plt.title('SBP for AA')
plt.xlabel("Values of SBP")

plt.subplot(336)
plt.hist(y2)
plt.title('SBP for CC')
plt.xlabel("Values of SBP")

# 3. Three Box plot
plt.subplot(337)
plt.boxplot(y)
plt.title("Boxplot of SBP")

# two box plot together
plt.subplot(338)
plt.boxplot(y1)
plt.title('Box Plot of SBP for AA')

plt.subplot(339)
plt.boxplot(y2)
plt.title('Box Plot of SBP for CC')
plt.tight_layout()


# ------------
# Q25
# ------------
import matplotlib.pyplot as plot

# method 1, by selecting random rows
# create a random of 50 observations
random_subset = d2.sample(n = 50, replace = False)
random_subset

# stem-leaf plot
# ------------
# Q26
# ------------
k = round(random_subset.SBP)
k
y = pd.Series(k)
plot.stem(y)

import stemgraphic
fig, ax = stemgraphic.stem_graphic(y)

# ------------
# Q27
# ------------
#subsetting data
#subset the d1 dataframe by excluding variables
exclude = ["ID","TC", "TG", "HDL", "LDL"]
keep = [var for var in d1.columns if var not in exclude]

d3 = d1[keep].copy() # subsetting
d3.head()
print(d1.columns)# we have already removed ID if you ran every line so far. Therefore, only 4 variables are removed
print(d3.columns)

# ------------
# Q28
# ------------

#Check the dimension of the new data d3. Report the number of missing values for each variable
np.shape(d3) #check for diamention
d3.isnull().sum() # reporting missing values

# putting it into a column. It will report number of missing values in each row
#remember axis = 1 for row and 0 for columns
d3["MISSING"] = d3.isnull().sum(axis=1)
d3.head()

d3["MISSING"].sum()

#------------
# Q29
#------------
#delete any missing values (rows) from d3, and denote this data as d4
d4=d3.dropna()

#------------
# Q30
#------------
#create  AGE1 variable in D4 data by rounding the AGE variable so that there will not be decimal
d4['AGE1'] = round(d4['AGE']).astype(int)
d4.head()

#------------
# Q31
#------------
#Subset the data by a sub interval of quantitative variable
#Subset the data again by keeping age from 9 to 20
d5 = d4.loc[(d4["AGE1"] >= 9) & (d4['AGE1']<=20)].copy()

#------------
# Q32
#------------
# exporting the dataframe d5 as csv format. You will find it in the same folder for python code file
# I mean working directory
d5.to_csv("d5_Cleaned_data.csv",index=False)

#------------
# Q33
#------------
d6 = d5.copy().groupby("AGE1") # Age is the column for groupby
d6.head()

# Note that d6 is a groupby dataframe object
# Let is see how it looks like

for AGE1, AGE1_d5 in d6:
    print(AGE1)
    print(AGE1_d5)

d6.get_group(15) # we will get data from age 15 dataframe

# Now you can run analytics on the group by
d6.max()
d6.min()
d6.describe()
d6.plot() # The graph is going to be messy as it contains all the data

# What we have done in this part can be summarized as split apply and combine
#------------
# Q34
#------------
# get number of observations by each age category
print("Observations by Age:")
print(d6.size()) # Here you are using groupby to get size

print("Total Observations:")
print(sum(d6.size())) # you sum the groupby result

print("Number of age categories")
len(d6)

#print group by age category of 11
d6.get_group(11).head()


for AGE1, AGE1_d5 in d6:
    print(AGE1)
    print(AGE1_d5)

# ------------------
# Q35 and Q36
# ------------------
# pseudocode -  Single looping
# ANALYZE the columns you need and subset the data
# DROP missing values or impute and create new data frame as Missing values may interfere with the calculations
# Again SUBSET the age group required to new dataFrame
# Now get the rounded value of age into new variable called NewAge
# Create a new list grouped by NewAge. This will create a dataframe under each list
# You will find that the length of the list will be same as number of categories
# Now iterate through each item of list, Within each dataframe, use ss library to find appropritate statistics
# Bind the list together by concatenate function.

import statistics as ss
datalist = [] # note that this is a list and not a dataframe


#subset the data by getting columns you need from original data d1
d8 = d1[['RACE','INCOME','AGE','SMOKE','SBP','DBP','HT', 'WT','WM','BMI']]
d9 = d8.dropna() # remove missing value rows

d11 = d9[(d9.AGE > 9) & (d9.AGE <= 19)] # subset the data by age from 10 to 19
d11['NewAge'] = round(d11.AGE,0) # now you round it
ds = [rows for _, rows in d11.groupby('NewAge')]
len(ds)
# NOTE: ds is a groupby object as described before

#I just wanted to see my list nicely to analyze the list. You could just use ds, and it will do it.
from pprint import pprint
pprint(ds)
ds[1] # good thing is you could get each dataframe by calling the appropriate number of list
d11.groupby("NewAge").mean()

len(ds)
for i in range(len(ds)):
       datalist.append(pd.DataFrame({
      'age':[ss.mean(ds[i].NewAge)],
      'Mean.SBP': [ss.mean(ds[i].SBP)],
      'Mean.DBP': [ss.mean(ds[i].DBP)],
      'Mean.HT': [ss.mean(ds[i].HT)],
      'Mean.WT': [ss.mean(ds[i].WT)],
      'Mean.BMI': [ss.mean(ds[i].BMI)]}))

pd.concat(datalist)

# Note that you get the groupby results - that is for each group you get summary statistics specified
pprint(datalist)
datalist[1]

# Great !!! Buhahaaha!!


# ------------
# Q37, Q38, Q39
# ------------
import statistics as ss
# Select 10 rows & 10 columns from D4,

# Without double loop
# we have already selected 10 columns individually. Now select 10 rows
D4 = d9[0 :10]

# Remember axis = 0 is for columns. Here we are selecting mean of means.
# Mean for ech column and mean for all column
# We are finding the grand mean of datum all together
G = ss.mean(D4.mean(axis=0))

D4 - G #substarct the grand mean from each datum
np.power((D4 - G), 2) # now we square it  -  perform the operations specified


# DOUBLE LOOP, How to write the double loop
# for this, we convert it into matrix format - we can iterate through the loop
d41 = D4.as_matrix()
#why range is 10 ? Because we have 10 rows and 10 columns
for i in range(10) :
    for j in range(10) :
        d41[i, j] = (d41[i, j] - G) ** 2 # from each element we remove G and square it
d41

# ------------
# Q40
# ------------
D1 = pd.read_csv("PROJECTDATA.csv")
# look data is longitudinal
# Longitudinal data, sometimes called panel data, is a collection of repeated observations of the same subjects,
# taken from a larger population, over a period of time – and is useful for measuring change.

# you cannot do regular regression if needed - look into mixed regression

# Objective of the question:
# Create NEWID variable from the existing ID variable in the D1 dataframe
# NEWID variable should only have numeric numbers. We can call this dataframe d8

del D1['Unnamed: 0'] # we do not need this column

# Make a deep copy, including a copy of the data and the indices.
# With deep=False neither the indices nor the data are copied.
D3 = D1.copy(deep=False)

D8 = D1.copy(deep=True)

D8['NEW.ID'] = D8['ID'] - 100010
del D8['ID']
D8.head()

# NOw you can see why this longitudinal

# ------------
# Q41
# ------------
# Sort the D8 data by column names
D8_column_list = D8.columns # get all column list
sorted_D8_columns = sorted(D8_column_list)# sort it

D8 = D8[sorted_D8_columns] # reset the dataframe by specified column list
D8.head() # Let us see how it looks like

# ------------
# Q42
# ------------
#To wide format
# Note that our dataframe is in longitudinal format
# Let us convert it to wide format
# creating SUBSET of data
D8 = D8[['NEW.ID', 'VISIT', 'SBP']] # Create subset data by selecting variables needed

# Goal is to get ID as rows and Visit as column
D8['VISIT'].loc[D8['VISIT'] == 0] = 1

# Widening -  we get SBP value for different visit
W1 = D8.pivot(index="NEW.ID", columns='VISIT').sort_index(axis=1, level=1).sort_index().copy(deep=True)
W1

# ------------
# Q43
# ------------
#From wide format --- Back to Long format
L1 = W1.copy().stack(level=1).reset_index(level=1)
sorted_L1_columns = sorted(L1.columns)
L1 = L1[sorted_L1_columns]
L1


# ------------
# Q44
# ------------
# Entering 10% missing values

columns = ['AGE', 'SBP', 'DBP', 'HT', 'WT', 'BMI']
column_list_length = len(columns)
D2 = D1[columns].copy().dropna() # delete all rows that has missing values

num_observations = len(D2)
sample_size = round(num_observations * .10)


def add_missingness(variable, current_loop_number, n) :
    sample = variable.sample(n, replace=False, random_state=current_loop_number)
    variable.loc[variable.index.isin(sample.index)] = np.nan
    return variable


for i in range(column_list_length) :
    variable = columns[i]
    D2[variable] = add_missingness(D2[variable], current_loop_number=i, n=sample_size)

# Display the resulting changes to a subset of the variables:
print(D2[D2['AGE'].isnull()].head())


def calculate_null_proportion(variable) :
    return pd.isnull(variable).sum() / len(variable)

# Take a look at percent of null values
print("Null Proportions:")
D2.apply(calculate_null_proportion) # Note this method to apply function to Dataframe

# ------------
# Q45
# ------------
# Again subsetting
columns = ["ID", "AGE", "SBP", "DBP"]
N1 = D1[columns].copy()

# ------------
# Q46
# ------------
# Again subsetting
columns = ["ID", "TC", "TG", "HT", "WT"]
N2 = D1[columns].copy()
N2

# ------------
# Q47
# ------------
# Merge N1 and N2 data side by side in such a way that the merged data should have only one ID variable
N_merged = N1.merge(N2, left_on='ID', right_on='ID') # Merge by ID variable
N_merged.head()

# ------------
# Q48
# ------------
# subsetting data by taking a random sample of 1000 rows from the first 10000 rows
M1 = D1.iloc[:10000].sample(1000, replace=False, random_state=2)
M1

# ------------
# Q49
# ------------
# Subset data by taking a random sample of 1000 rows from the remaining
M2 = D1.loc[~D1.index.isin(M1.index)].sample(1000, replace=False, random_state=3)
# The "~" symbol states to get everything where the following statement is NOT true (negation)
M2

# ------------
# Q50
# ------------
# Merge M1 with M2 in such a way that rows from M2 lies beneath the rows from M1
M_merged = M1.append(M2)
M_merged

len(initial_bp_variables)




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


# End of Time Variant


# Python Code:
# --------------
# Part III
# --------------
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import math
import statistics as ss
import scipy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import scipy.stats as stats
import statsmodels.stats as sms
from statsmodels.stats import api as sms_api
from textwrap import fill


d1 = pd.read_csv('C:/Users/HOMEPCUSER/Desktop/Spring 19/PythonProgramming/PROJECTDATA.csv')
d8 = d1[['RACE', 'INCOME', 'AGE', 'SMOKE', 'SBP', 'DBP', 'HT', 'WT', 'WM', 'BMI']]
d9 = d1.dropna()

# Q1
# one sample t-test.
# These codes are for two sided hypothesis test.

# Testing if mean is certain value or not
# 1. For the variables AGE, SBP and WT test whether age average is 15 years or not,
scipy.stats.ttest_1samp(d9.AGE, 15)

# test whether SBP average is 100 or not
scipy.stats.ttest_1samp(d9.SBP, 100)
scipy.stats.ttest_1samp(d9.DBP, 85)
scipy.stats.ttest_1samp(d9.BMI, 22)

# test whether WT average is 65 or not. These are both tail alternative hypothesis.
scipy.stats.ttest_1samp(d9.WT, 65)


# 2.	Repeat 1 where alternative hypothesis is right tailed.
# Do it by dividing alpha.

# 3.	Test the equality of the two means for variables AGE, SBP, DBP, WT, BMI, TC when RACE=1 and RACE=2.
scipy.stats.ttest_ind(d9[d9.RACE == 1].SBP, d9[d9.RACE == 2].SBP)
scipy.stats.ttest_ind(d9[d9.RACE == 1].AGE, d9[d9.RACE == 2].AGE)
scipy.stats.ttest_ind(d9[d9.RACE == 1].DBP, d9[d9.RACE == 2].DBP)
scipy.stats.ttest_ind(d9[d9.RACE == 1].WT, d9[d9.RACE == 2].WT)
scipy.stats.ttest_ind(d9[d9.RACE == 1].BMI, d9[d9.RACE == 2].BMI)
scipy.stats.ttest_ind(d9[d9.RACE == 1].HT, d9[d9.RACE == 2].HT)

# Q4
# Test on the equality of proportion for two categories

# Test the equality of proportion for smoking when RACE=1 and RACE=2.
# A hypothesis test formally tests if the proportion of smoking in RACE1 and RACE2 populations are equal.

# When one variable is an explanatory variable (X, fixed) and the other a response variable (Y, random),
# the hypothesis of interest is whether the populations have the same or different proportions in each category.
pd.crosstab(d9.RACE, d9.SMOKE)

a = pd.crosstab(d9[d9.RACE == 1].SMOKE, columns="counts")
b = pd.crosstab(d9[d9.RACE == 2].SMOKE, columns="counts")
print(a)
print(b)

# add value like a[1], a[0], b[1], b[0]
aa = np.array([[3603, 216], [4047, 82]])
print(stats.chi2_contingency(aa))  # chi-squared value and p value

# Test method. Use the chi-square goodness of fit test to determine whether observed sample frequencies
# differ significantly from expected frequencies specified in the null hypothesis.

# Q5
# Construct a confidence interval of mean for SBP and TC.

import statsmodels.stats.api as sms
sms.DescrStatsW(d9.AGE).tconfint_mean()  # Confidence interval of mean for AGE -  default confidence level is 95%
sms.DescrStatsW(d9.SBP).tconfint_mean()  # Confidence interval of mean for SBP -  default confidence level is 95%


# alternative approach for the same problem -  you could use this for different confidence level
stats.t.interval(0.95, len(d9.AGE)-1, loc=np.mean(d9.AGE), scale=stats.sem(d9.AGE))
stats.t.interval(0.95, len(d9.SBP)-1, loc=np.mean(d9.SBP), scale=stats.sem(d9.SBP))

# alternative approach
sbp_confidence_interval = sms.DescrStatsW(d1['SBP'].dropna()).tconfint_mean()
bmi_confidence_interval = sms.DescrStatsW(d1['BMI'].dropna()).tconfint_mean()
print(sbp_confidence_interval)
print(bmi_confidence_interval)


# Q6

# Confidence interval for the difference of Mean
# Construct a confidence interval for the difference of mean for SBP when RACE=1 and RACE=2.
race_1_data = d1.loc[(d1['RACE'] == 1) & (pd.notnull(d1['RACE']))]
race_2_data = d1.loc[(d1['RACE'] == 2) & (pd.notnull(d1['RACE']))]

test_variables = ['AGE', 'SBP', 'DBP', 'WT', 'BMI', 'TC']

# if you want to see the summary of new dataset
race_1_stats = race_1_data[test_variables].describe()
race_2_stats = race_2_data[test_variables].describe()

# Objective is to Construct a confidence interval for the difference of mean for SBP for both race
# so we drop the missing value of SBP
sbp_race_1_stats_obj = sms.DescrStatsW(race_1_data['SBP'].dropna())
sbp_race_2_stats_obj = sms.DescrStatsW(race_2_data['SBP'].dropna())

sbp_mean_comparison_obj = sms_api.CompareMeans(sbp_race_1_stats_obj, sbp_race_2_stats_obj)
ci_for_diff_btw_mean = sbp_mean_comparison_obj.tconfint_diff()
print(ci_for_diff_btw_mean)

# Q7
# Construct a confidence interval for proportion of smokers -  categorical variable

table = pd.crosstab(d9['SMOKE'], columns='count')
print(table)

import statsmodels.stats.proportion as one
ci_low, ci_upp = one.proportion_confint(74, 1868, alpha=0.05, method='normal')
print(ci_low, ci_upp)

# Q8
# Also construct a confidence interval for difference of proportions for smokers when RACE=1 and RACE=2.
# Two sample proportion confidence interval

smoker_count = len(d1.loc[d1['SMOKE'] == 1])
print(smoker_count) # get the number of smoker

total_count = len(d1['SMOKE'].dropna())
print(total_count) # get the total number of sample
mean_prop = smoker_count / total_count # obtain the mean


def two_sample_prop_ci(sample1_success, sample1_total, sample2_success, sample2_total, alpha=.05) :
    s1_prop = sample1_success / sample1_total
    s2_prop = sample2_success / sample2_total
    prop_mean_difference = s1_prop - s2_prop

    s1_std = np.sqrt(s1_prop * (1 - s1_prop) / sample1_total)
    s2_std = np.sqrt(s2_prop * (1 - s2_prop) / sample2_total)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    margin_of_error = z_crit * (s1_std + s2_std)

    lower_limit = prop_mean_difference - margin_of_error
    upper_limit = prop_mean_difference + margin_of_error
    return lower_limit, upper_limit

# ----------------------------------------
# confidence interval for smoker
# ----------------------------------------

smoker_confidence_int = sms_api.proportion_confint(smoker_count,total_count)

print("Confidence interval for the proportion of patients who smoke:")
print(smoker_confidence_int)


# ------------------------------------------------------------------
# confidence interval for the difference of smokers by RACE
# ------------------------------------------------------------------
# Note we already use the previously created race_1_data
race_1_smoker_count = len(race_1_data.loc[race_1_data['SMOKE'] == 1])
race_1_total_count = len(race_1_data['SMOKE'].dropna())

race_2_smoker_count = len(race_2_data.loc[race_2_data['SMOKE'] == 1])
race_2_total_count = len(race_2_data['SMOKE'].dropna())


smoker_ci_by_race = two_sample_prop_ci(race_1_smoker_count, race_1_total_count,race_2_smoker_count, race_2_total_count)
print("Confidence interval for the mean difference of smokers within Race 1 in comparison")
print("proportion of patients who smoke:")
smoker_ci_by_race


# Q9
# A paired t-test is used to compare two population means where you have two samples in
# which observations in one sample can be paired with observations in the other sample.

# Conduct a pairwise t test for SBP and DBP.
stats.ttest_rel(d9['SBP'], d9['DBP'])


# Q10
# Chi-square test of variance
# A chi-square test can be used to test if the variance of a population is equal
# to a specified value. This test can be either a two-sided test or a one-sided test.
# The two-sided version tests against the alternative that the true variance is either less than
# or greater than the specified value. The one-sided version only tests in one direction.
# The choice of a two-sided or one-sided test is determined by the problem.

# Test whether variance of DBP is 60 or not.

def chi_sq_test_for_variance(variable, h0) :  # function for the chi squared distribution associated with the above formula

    sample_variance = variable.var()  # Find the variance of the sample
    n = variable.notnull().sum()  # Take the sum of the number of values that are not missing
    # the actual number of observations for the variable where
    # True = 1, False = 0
    degrees_of_freedom = n - 1  # Find the degrees of freedom
    x_sq_stat = (n - 1) * sample_variance / h0  # Using the formula above to calculate the X^2 statistic
    p = stats.chi2.cdf(x_sq_stat, degrees_of_freedom)  # Here, a cumulative distribution function is used to determine
    # the significance of the variance using the X^2 statistic.

    # If a chi square test statistic is over the 99th percentile,
    # we'd have reason to suspect significance at alpha = .05.
    # We need to account for circumstance where the p value is greater
    # than .05, however:
    if p > .5 :
        p = 1 - p
    return (x_sq_stat, p, degrees_of_freedom)  # End of function

dbp_variance = round(d9["DBP"].var(), 2)
x_sq_stat, pval, dof = chi_sq_test_for_variance(d9["DBP"], h0=60)
print('Chi-square statistic, pvalue, degrees of freedom')
print(x_sq_stat, pval, dof )

# Q11
# Test on equality of variances for the variables SBP & DBP.

import statistics as stats
import scipy.stats as ss


def Ftest_pvalue(d1, d2) :
    df1 = len(d1) - 1
    df2 = len(d2) - 1
    F = stats.variance(d1) / stats.variance(d2)
    single_tailed_pval = ss.f.cdf(F, df1, df2)
    double_tailed_pval = single_tailed_pval * 2
    return (single_tailed_pval, double_tailed_pval)


k = Ftest_pvalue(d9["SBP"], d9["DBP"])

print('Single tailed p-value, Double tailed p-value')
print(k)

# Q12
# Non- parametric test of median -  test whether median is specified value or not
# Test whether median SBP is 100 or not. This is nonparametric test.
# The one-sample Wilcoxon signed rank test is a non-parametric alternative to one-sample t-test when
# the data cannot be assumed to be normally distributed. It’s used to determine whether the median of
# the sample is equal to a known standard value

from scipy.stats import ttest_1samp, wilcoxon, ttest_ind, mannwhitneyu

z_statistic, p_value = wilcoxon(d9.SBP - 100)
print
"one-sample wilcoxon-test", p_value

# Q13
#  the Mann–Whitney U test is a nonparametric test that is used to compare two sample means that come from
#  the same population, and used to test whether two sample means are equal or not.  Usually, the Mann-Whitney U test
#  is used when the data is ordinal or when the assumptions of the t-test are not met.

# Assumptions of the Mann-Whitney:
#
# Mann-Whitney U test is a non-parametric test, so it does not assume any assumptions related to the
# distribution of scores.  There are, however, some assumptions that are assumed
# 1. The sample drawn from the population is random.
# 2. Independence within the samples and mutual independence is assumed.  That means that an observation
# is in one group or the other (it cannot be in both).
# 3. Ordinal measurement scale is assumed.

# Test equality of median for SBP corresponding to RACE variable. This is nonparametric test.
u, p_value = mannwhitneyu(race_1_data.SBP, race_2_data.SBP)
print(u, p_value)

# Q14
# Test whether median SBP and median DBP are equal or not.
u1, p_value1 = mannwhitneyu(d9.SBP, d9.DBP)

print(u1, p_value1)

# Q15
# In the MWW test you are interested in the difference between two independent populations
# (null hypothesis: the same, alternative: there is a difference) while in Wilcoxon signed-rank test
# you are interested in testing the same hypothesis but with paired/matched samples.

# nonparametric alternative to Pair t test for testing null hypothesis of zero difference between SBP and DBP

z_statistic, p_value = wilcoxon(d9.SBP - d9.DBP)
print(z_statistic, p_value)

# getting a zero z statistic
# The z value that we obtain indicates how many SE the observed value differs from the expected value.
# We can then use the Normal Table to attach a percentile to this z score.
#
# The percentile is referred to as P, and is the probability that we would observed this value or farther away
# from the expected value if in fact the expected value were true and what we are looking at is simply chance variation.

# Q16

# Create a new CSBP variable with labels 1, 2, and 3 by categorizing SBP variable. 1 stands for SBP values
# less than 85 (inclusive), 2 takes values greater than 85 and less than 115 and 3 stands for SBP values greater
# than 115. Similarly create a new CWT variable with labels 1, 2, and 3 by categorizing WT variable. 1 stands for
# weights less than 44 (inclusive), 2 stands for weights greater than 44 and less than 55 and 3 stands for
# weights greater than 55. Conduct a chi square test of independence for CWT and CSBP variables.

# split SBP into category
d9['CSBP'] = pd.cut(d9.SBP,
                     bins=[0, 85, 115, 400],
                     labels=["1", "2", "3"])
# split WT variables into category
d9['CWT'] = pd.cut(d9.WT,
                     bins=[0, 44, 55, 300],
                     labels=["short", "average", "tall"])

a1 = pd.crosstab(d9.CSBP, columns="counts")
print(a1)
b1 = pd.crosstab(d9.CWT, columns="counts")
print(b1)
# get value from a1 and b1
bb=np.array([[17,1650,275],[480,705,757]])
print(bb)
scipy.stats.chi2_contingency(bb)

# Look how functions are created in python
def f(x):
    y = math.pow(x, 2)+math.exp(-x)+math.log(5*x)+2
    return y


#---------------------------------------------------------------
#   Correlation, Regression and Anova
#---------------------------------------------------------------


import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import scipy.stats as stats
import statsmodels.stats as sms
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats import api as sms_api
from statsmodels.graphics.gofplots import ProbPlot
from sklearn import metrics
from textwrap import fill  # for the sake of automatically creating newlines after n characters when writing paragraphs.
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison   ## ANOVA POST HOC ##
from patsy import dmatrices  ## design matrices for Negative Binomial Regression

# --------------
# Part 4
# --------------
d1 = pd.read_csv('C:/Users/HOMEPCUSER/Desktop/Spring 19/PythonProgramming/PROJECTDATA.csv')
d2 = d1.dropna()

# Q1
# Test whether two quantitative variables are correlated or not
stats.pearsonr(d2.SBP, d2.DBP)
stats.pearsonr(d2.HT, d2.WT)
stats.pearsonr(d2.HT, d2.WT)
1
# Q2
# Here we construct a correlation matrix on a set of quantitative variables with the p values
cv = ["SBP", "DBP", "HT", "WT", "WM", "BMI", "TC", "TG", "HDL","LDL"]
d2[cv].corr()
import seaborn as sns
stats.pearsonr(d2["SBP"], d2["WT"])
sns.heatmap(d2[cv].corr())
sns.heatmap(d2[cv].corr(), cmap='BuGn')


# ---------------------------------------------
# correlation matrix with p values
# ---------------------------------------------
import numpy as np
import pandas as pd
from scipy.stats import pearsonr # for correlation matrix p - values

df1 = pd.DataFrame(d2[cv])
df2 = pd.DataFrame(d2[cv])
coeffmat = np.zeros((df1.shape[1], df2.shape[1]))
pvalmat = np.zeros((df1.shape[1], df2.shape[1]))

for i in range(df1.shape[1]):
    for j in range(df2.shape[1]):
        corrtest = pearsonr(df1[df1.columns[i]], df2[df2.columns[j]])
        coeffmat[i,j] = corrtest[0]
        pvalmat[i,j] = corrtest[1]

dfcoeff = pd.DataFrame(coeffmat, columns=df2.columns, index=df1.columns)
print(dfcoeff)
dfpvals = pd.DataFrame(pvalmat, columns=df2.columns, index=df1.columns)
print(dfpvals)

# Q3
# using  both   Pearson   correlation   coefficients   and   Spearman correlation coefficient
stats.spearmanr(d2.SBP, d2.DBP)
stats.spearmanr(d2.HT, d2.WT)
stats.spearmanr(d2.HT, d2.WT)
stats.spearmanr(d2[cv])

# -----------------------------------------
# correlation matrix with p values
# -----------------------------------------

df1 = pd.DataFrame(d2[cv])
df2 = pd.DataFrame(d2[cv])
coeffmat = np.zeros((df1.shape[1], df2.shape[1]))
pvalmat = np.zeros((df1.shape[1], df2.shape[1]))

for i in range(df1.shape[1]):
    for j in range(df2.shape[1]):
        corrtest = stats.pearsonr(df1[df1.columns[i]], df2[df2.columns[j]])
        coeffmat[i,j] = corrtest[0]
        pvalmat[i,j] = corrtest[1]

dfcoeff = pd.DataFrame(coeffmat, columns=df2.columns, index=df1.columns)
print(dfcoeff)
dfpvals = pd.DataFrame(pvalmat, columns=df2.columns, index=df1.columns)
print(dfpvals)

# Q4
# Test for normality
# The null hypothesis for this test is that the data are normally distributed.
stats.shapiro(d2['SBP'])

# Q5
# Log transformation
log_sbp = np.log(d2['SBP'])
stats.shapiro(log_sbp)

# Q6
# Construction of  normal probability plot (QQ plot for two quantitative variable)
import pylab
plt.subplot(221)
stats.probplot(d2['SBP'], dist="norm", plot=pylab)
pylab.show()
plt.subplot(222)
stats.probplot(log_sbp, dist="norm", plot=pylab)
pylab.show()

# Q7
# ply Box- Cox transforman on quantitative variable

d3, labmda = stats.boxcox(d2.SBP)

import pylab
plt.subplot(331)
stats.probplot(d2['SBP'], dist="norm", plot=pylab)
pylab.show()
plt.subplot(332)
stats.probplot(log_sbp, dist="norm", plot=pylab)
pylab.show()
plt.subplot(333)
stats.probplot(d3, dist="norm", plot=pylab)
pylab.show()

# normality test on box cox transformed sbp
stats.shapiro(d3)

# Q8
# Simple linear regression
# Add intercept manually
d3 = d1[["SBP","WT"]].dropna()
d3["INT"] = 1
rm = statsmodels.regression.linear_model.OLS(d3['SBP'].values,d3[["INT",'WT']].values,intercept=True).fit()
predictions=rm.fittedvalues
print(rm.summary())
intercept,wt_coef = rm.params

# without 1 column no intercept output
a=sm.OLS(d2.SBP, d2.HT).fit()
a.summary()

# Q9
# Repeat question 8 by using logarithm of SBP and WT as independent
rml= statsmodels.regression.linear_model.OLS(np.log(d3['SBP'].values),d3[["INT",'WT']].values,intercept=True).fit()
predictions=rml.fittedvalues
print(rml.summary())
intercept,wt_coef = rml.params

# Q10
# Repeating regression using boxcox transformed variable
Y = stats.boxcox(d3.SBP)
d3["BCSBP"]=Y[0]
rml= statsmodels.regression.linear_model.OLS(d3['BCSBP'].values,d3[["INT",'WT']].values,intercept=True).fit()

# Q11
# Multiple linear regression
d3 = d1[["SBP","AGE","WT","TC","BMI"]].dropna()
d3["INT"] = 1
rml1= statsmodels.regression.linear_model.OLS(np.log(d3['SBP'].values),d3[['INT','AGE','WT','TC','BMI']].values,intercept=True).fit()

# Q12
# ANOVA model to test the equality of quantitative variable means (BMI) for the different categories of
# categorical variable
from statsmodels.formula.api import ols
d3 = d1[["WM","BMI"]].dropna()

model_name = ols('BMI ~ WM', data=d3).fit()
model_name.summary()

# Q13 -  Logistic regression
# only 0, 1 is acceptable
d3=d1[["RACE","AGE", "SBP", "TG", "TG", "LDL"]].dropna()
d44=d3.RACE.replace(to_replace=2, value = 0) # changing label 2 to 0
d3["Race"]=d44
import statsmodels.api as sm
logit_model=sm.Logit(d3.Race,d3.SBP)
result=logit_model.fit()
print(result.summary2())


#-----------------------------------------------------------------
# Take Home Regression test
#-----------------------------------------------------------------

# Import standard packages
import numpy as np
import pandas as pd

# additional packages
from lifelines import CoxPHFitter
from scipy import stats
from statsmodels.formula.api import glm
from statsmodels.genmod.families import Binomial
import statsmodels.api as sm

# Any issues installing packages can be avoided by installing dependencies.
# If there is any issue with installing dependencies, use windows binaries as alternatives

# Q1
# reading csv data
d1 = pd.read_csv('C:/Users/HOMEPCUSER/Desktop/Spring 19/PythonProgramming/PROJECTDATA.csv')
print(d1)

d2 = d1.dropna() # drop missing variables
print(d2)
d3 = d1.dropna()

d1["RACE"].describe()
d2["RACE"].describe()

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_d1_1 = LabelEncoder()
d2.loc[:, "RACE"] = labelencoder_d1_1.fit_transform(d2.RACE)

# fit the model

# --- >>> START stats <<< ---
model = glm('RACE ~ AGE+ SBP+ TC+ TG+LDL', data=d2, family=Binomial()).fit()
# --- >>> STOP stats <<< ---

print(model.summary())


# Q2

#Creating categorical variable from numerical

# Because we will be binning the data using inclusive right binning functions,
# We make sure that the highest value is still included in the new categorical
# Label when creating bins
CSBP_ranges = [0,85,115,np.inf] # The bins will be as follows: (0,85],(85,`115], (100, infinity)

# the labels of the new categprical variable
CSBP_labels = [1,2,3]

# This function creates categorical variables out of quantitative
# Note:  Bins includes the rightmost edge
d2['CSBP'] = pd.cut(d2['SBP'].values, bins = CSBP_ranges, right = True, labels = CSBP_labels, include_lowest = False )
d2.head()


# fit the model
# split the data in dependent and independent variables
Xvariables = ["AGE", "TG", "RACE","LDL"]
X = d2[Xvariables]
Y = d2["CSBP"]
Xc = sm.add_constant(X)

#instantiate and fit multinomial logit
mlogit = sm.MNLogit(Y, Xc)
fmlogit = mlogit.fit()
print(fmlogit.summary())



# Q3

# fit the model
variables = ["AGE", "BMI",  "WT", "SUGAR"]
X = d2[variables]
Y = d2["NLOG"]
Xc = sm.add_constant(X)


model = sm.GLM(Y, Xc, family=sm.families.Poisson(link=None)).fit()
print(model.summary())


# Q4
newvariables = ["AGE", "SBP", "BMI", "TC", "TIME","EVENT"]
mydataset = d2[newvariables]

fitter = CoxPHFitter()
fitter.fit(mydataset, duration_col='TIME', event_col='EVENT', show_progress=True)
fitter.print_summary()


# Q5
stats.kruskal(d2.SBP, d2.INCOME)

#----------------------------------------------------------------
# Playing with Python functions
#----------------------------------------------------------------

# SIDE NOTE:  REMEMBER TO KEEP INDENTATION IN PYTHON FUNCTION -  NO EXCUSE
# Q1

# Function to get maximum of two numbers
def mymax2(x, y) :
    """Return larger of x and y."""
    largest_so_far = x
    if y > largest_so_far :
        largest_so_far = y
    return largest_so_far

# Function to take inputs
def main() :
    print("Mymax:Enter two values to find the larger.")
    first = float(input("First value:"))
    second = float(input("Second value:"))
    print("The larger value is", mymax2(first, second))

main()


# alternative approach
def mymax2(x, y) :
    """Return larger of x and y."""
    if x > y :
        return x
    else :
        return y

print("max value of 22 and 2 is :", mymax2(22,3))

# write a program for minimum value

def mymin2(x, y) :
    """Return smaller of x and y."""
    if x < y :
        return x
    else :
        return y

print("min value of 22 and 2 is :", mymin2(22,3))

# Function to return  absolute value of x
def myabs(x) :
    """Return absolute value of x."""
    if x >= 0 :
        return x
    else :
        return -x

print("Absolute Value of -10 is :", myabs(-10))

# Q3 continuous looping for different values of x and y
# Double looping in function
def main() :
    for x in range(10) :
        for y in range(10) :
            print("mymax2(", x, ",", y, ") =", mymax2(x, y))

main()


# Q4 finding smallest of three values
def mymin3(x, y, z) :
    """Return smaller of x, y, and z."""
    smallest_so_far = x
    if y < smallest_so_far :
        smallest_so_far = y
    if z < smallest_so_far :
        smallest_so_far = z
    return smallest_so_far


def main() :
    print("Mymax:Enter three values to find the larger.")
    first = float(input("First value:"))
    second = float(input("Second value:"))
    third = float(input("Third value:"))
    print("The smaller value is", mymin3(first, second, third))


main()


#  Function to find Middle value
def median3(x, y, z) :
    """Return the median of x, y, and z."""
    if (y <= x and x <= z) or (z <= x and x <= y) :
        return x
    elif (x <= y and y <= z) or (z <= y and y <= x) :
        return y
    else :
        return z

median3(12, 30, 22)

# Q6 letter grade from points
def grade(score) :
    """Return the letter grade equivalent of score."""
    if score >= 90 :
        return "A"
    elif score >= 80 :
        return "B"
    elif score >= 70 :
        return "C"
    elif score >= 60 :
        return "D"
    else :
        return "F"

grade(45)

# Q7
# COMPOUNT INTEREST ANNUAL
# Account balance with interest
# principal=40
# interest=.20
# years=50

def balance(p, r, t) :
    """Return new balance using compound annual interest."""
    return p * (1 + r) ** t


def main() :
    print("calculate compound interest over time.")
    principal = float(input("principal:"))
    rate = float(input("Interest rate(as a decimal):"))
    years = int(input("Number of years: "))
    for year in range(years + 1) :
        print(year, (balance(principal, rate, year)))


main()


# Q8 Different formula
# COMPOUNT INTEREST MONTHLY
def compound(p, r, t) :
    """Return new balance with interest compounded monthly."""
    return p * (1 + r / 12) ** (12 * t)

def main() :
    print("Calculates compound interest over time.")
    principal = float(input("Principal: "))
    rate = float(input("Interest rate (as a decimal): "))
    years = int(input("Number of years: "))
    for year in range(years + 1) :
      print(year, compound(principal, rate, year))

main()


# Q9
#  COMPOUNT INTEREST BASED ON COMPOUNTING PERIOD PER YEAR
# Number of period is n
def compound(p, r, t, n) :
    """Return new balance using compound interest."""
    return p * (1 + r / n) ** (n * t)


def main() :
    print("Calculates compound interest over time.")
    principal = float(input("Principal: "))
    rate = float(input("Interest rate (as a decimal): "))
    years = int(input("Number of years: "))
    periods = int(input("Number of compounding periods per year: "))
    for year in range(years + 1) :
        print(year, compound(principal, rate, year, periods))


main()


# Function to calculate all 10 powers of 2
# Here n goes from 1 through 11
# for loop with specified range of i
def cnn() :
    for n in range(1, 11) :
        print(n, 2 ** n)
cnn()


# Q11
import numpy as np

# for loop where i increment as specified within a range from 10 to 201
def main() :
    for n in range(10, 201, 10) :
        print(n, np.log(n), np.exp(n), n ** 2, 2 ** n)

main()

# Q12

# Area of circle
def main() :
    for r in range(1, 10) :
        print(r, (np.pi) * r ** 2)

main()

# Q13 Accumulation loops/Accumulating interest

# Accumulating compound interest
def balance_accum(principal, rate, years) :
    """Return balance with accumulated compound interest."""
    balance = principal
    for _ in range(years) :
        interest = balance * rate
        balance += interest
    return balance

balance_accum(30, 1, 5)

# Q14 Some looping
# If you underscore the integer part of for loop,i starts with  1
result = 0
for _ in range(5) :
    result += 1
print(result)

# for loop that starts from 0 to n-1
# i starts with 0 and loop until i = n-1
result = 0
for i in range(5) :
    result += i
print(result)

# Loop interated 1 through 5 incrementing 2 at each interation
#  Here loop is working 5 times from 1 to 5
result = 0
for _ in range(5) :
   result += 2
print(result)

# for loop is working 5 times from 0 to 4
result = 0
for i in range(5) :
   result += 2 * i
print(result)

#for each i  loop increment result by 2
result = 0
for i in range(5) :
    print(i, result)
    result += 2

#j is jumping by 2 until 9, and adding to result -- Interesting result

result = 0
for j in range(1, 10, 2) :
    result += j
    print(j, result)


# loop goes from 0 to n-1
result = 0
for k in range(5) :
    print(k, result)
    result += 2 * k + 1

# Find some summation of the series
result = 0
for n in range(1, 101) :
    result += n
    print(n, result)

# Q16
result = 0
for n in range(5, 101, 5) :
    result += n
    print(n, result)

# Q17
result = 0
for n in range(1, 101) :
    result += 1 / n
    print(result)

# Q18
result = 0
for n in range(2, 101, 2) :
     result += n
     print(n, result)

result = 0
for n in range(1, 11) :
   result += n ** 2
   print(n, result)

result = 0
for n in range(1, 11) :
   result += 1 / n ** 2
   print(n, result)

# Loop for calculating factorial n
result = 1
for n in range(1, 20) :
   result *= n
    print(n, result)

result = 1
for n in range(1, 21) :
   result *= 2
print(result)

# harmonis sum
def harmonic(n) :
   """Return nth harmonic number."""
   total = 0
   for k in range(1, n + 1) :
      total += 1 / k
   return total

def main() :
   for n in range(1, 100) :
        print(n, harmonic(n))
main()

# harmonic sum
result = 0
for i in range(1, 10000) :
    result += 1.0 / i
    print(i, result)

result = 0
for i in range(1, 10000) :
    result += 1.0 / (i * i)
    print(i, result)

# saving account maturity year, page-59
def years_to_goal(principal, rate, goal) :
        """Return number of years to reach savings goal."""
        balance = principal
        years = 0
        while balance < goal :
            interest = balance * rate
            balance += interest
            years += 1
        return years

def main() :
        print("Calculates number of years to reach goal.")
        principal = float(input("Principal: "))
        rate = float(input("Interest rate (as a decimal): "))
        goal = float(input("Desired balance: "))
        print("The goal is reached in",
              years_to_goal(principal, rate, goal), "years.")

main()

# section 2.4 does the while loop
# will conver in nextThursday
# start tomorrow
def mysum(items) :
        """Return sum of values in items."""
    total = 0
    for item in items :
            total += item
    return total

def main() :
       data = [4, 9, 2, 7, 4, 8, 23, 54, 67, 90]
        print("My sum of", data, "is", mysum(data))

main()

#
def years_to_goal(principal, rate, deposit, goal) :
    """Return number of years to reach savings goal with deposits."""
    balance = principal
    years = 0
    while balance < goal :
            interest = balance * rate
            balance += interest
            balance += deposit
            years += 1
            return years

def main() :
        print("Calculates number of years to reach goal.")
        principal = float(input("Principal: "))
        rate = float(input("Interest rate (as a decimal): "))
        deposit = float(input("Annual additional deposit: "))
        goal = float(input("Desired balance: "))
        print("The goal is reached in", years_to_goal(principal, rate, deposit, goal), "years.")


# New project

from random import randint


def userguess(secret) :
    """Ask user for guesses until matching secret."""
    guess = int(input("Your guess"))
    while guess != secret :
        guess = int(input("Your guess?"))


def main() :
    secret = randint(1, 10)
    userguess(secret)


main()


# Q2
def present_value(c, r, t) :
    """Return present value."""
    return c / (1 + r) ** t

pv = present_value(c, r, t)


# Q3 repeated computaton of circle, with stop when -1
def circle_area(r) :
    """Return area of circle of radius r."""
    return np.pi * r ** 2


def ask_user() :
    """Get user input."""
    r = float(input("Radius (-1 to exit): "))
    while r != -1 :
        area = circle_area(r)
        print("The area is", area)
        r = float(input("Radius (-1 to exit): "))


def main() :
    print("This program will calculate the area of circles.")


ask_user()
print("Thank you.")


# Q4 repeated computation of volumnof spheres
def sphere_volume(r) :
    """Return volume of sphere of radius r."""
    return (4 / 3) * np.pi * r ** 3


def ask_user() :
    """Get user input."""
    r = float(input("Radius (-1 to exit): "))
    while r != -1 :
        volume = sphere_volume(r)
        print("The volume is", volume)
        r = float(input("Radius (-1 to exit): "))


def main() :
    print("This program will calculate the volume of spheres.")
    ask_user()
    print("Thank you.")


# Q5  Mile to kilometer conversion
def m_to_km(m) :
    """Convert distance in miles to kilometers."""
    return 1.609 * m


def ask_user() :
    """Get user input."""
    miles = float(input("Number of miles (-1 to exit): "))
    while miles != -1 :
        km = m_to_km(miles)
        print("That is", km, "kilometers")
        miles = float(input("Number of miles (-1 to exit): "))


def main() :
    print("This program will perform mile-to-kilometer conversions.")
    ask_user()
    print("Thank you.")


# Q5 compute the heart rate by using a formula
# Function to return estimate of maximum heart rate for person age years old
def max_heart_rate(age) :
    """Return estimate of maximum heart rate for person age years old."""
    return 208 - 0.7 * age

max_heart_rate(1)

def ask_user() :
    """Get user input."""
    age = float(input("Person’s age (-1 to exit): "))
    while age != -1 :
        rate = max_heart_rate(age)
        print("The estimated maximum heart rate is", rate)
        age = float(input("Person’s age (-1 to exit): "))
ask_user()

def main() :
    print("This program will calculate maximum heart rate estimates.")
    ask_user()
    print("Thank you.")

main()

# Q6 some python expression
list(range(6))

list(range(1, 20, 3))

[2 * i for i in range(5)]

[n ** 2 for n in range(8)]

list(range(2, 14))

list(range(10, 101, 10))

[2 * i + 1 for i in range(15, 20)]

[m ** 3 for m in range(1, 5)]

# write some python expression
list(range(1, 7))

list(range(2, 50, 4))

[0 * i for i in range(50)]  # 50 zeros

import random as rs
[rs.randint(0, 1) for _ in range(1000)]  # 1000 random 0 and 1

list(range(11, 27, 2))

list(range(72, 1, -3))

[n % 5 for n in range(50)]

[rs.randint(1, 6) for _ in range(1000)]



def mysum(items) :
    """Return sum of values in items."""
    total = 0
    for item in items :
        total += item
    return total

# Q9: sum of data  together with  commments
def main() :
        data = [4, 9, 2, 8, 3, 2, 5, 4, 2]
        print("Sum of", data, "is", sum(data))
        print("My sum of", data, "is", mysum(data))
main()


def main() :
    data = [rs.randint(-100, 101) for _ in range(500)]
    print("Sum of", data, "is", sum(data))
    print("My sum of", data, "is", mysum(data))

main()

# Function to return average of items
def mean(items) :
        """Return average of items."""
    return sum(items) / len(items)

def main() :
        data = [randint(1, 100) for _ in range(100)]
        print(data, mean(data))


# Function to find products of items
def product(items) :
    """Return product of items."""
    result = 1
    for item in items :
        result *= item
        return result

# Function to find geometric mean of items
def geometric_mean(items) :
    """Return geometric mean of items."""
    n = len(items)
    return product(items) ** (1 / n)


data = [4, 9, 2, 8, 3, 2, 5, 4, 2]


# Function to find sum of squares
def sum_of_squares(items) :
    """Return sum of squares of items."""
    return sum([item ** 2 for item in items])


# function to get RMS
def rms(items) :
    """Return root-mean-square of items."""
    return np.sqrt(sum_of_squares(items) / len(items))


# To get Even values list from a list
def evens(items) :
    """Return list of even values in items"""
    result = []
    for item in items :
        if item % 2 == 0 :
            result += [item]
    return result


# Swap function to change the position of the value in array
def swap(items, i, j) :
    """Swap items at indices i and j."""
    tmp = items[i]
    items[i] = items[j]
    items[j] = tmp


# function for target count and return  number of times target appears in items
def count(target, items) :
    """Return number of times target appears in items."""
    count = 0
    for item in items :
        if target == item :
            count += 1
    return count


# function to return random integer
def randints(a, b, n) :
    """Return n random integers between a and b."""
    return [randint(a, b) for _ in range(n)]





# ---------------------------------------------------------------
# Text Mining
# ---------------------------------------------------------------


# Function to Return conjugation of regular -ar Spanish verbs
# Removes last two letters and replace with post-fix
def conjugate(verb):
    """Return conjugation of regular -ar Spanish verbs."""
    stem = verb[:-2]
    return [stem + "o", stem + "as", stem + "a", stem + "amos", stem +"ais", stem + "an"]

def main():
    verb =input("Enter an -ar verb: ")
    print("Present indicative conjugation of " + verb + ":")
    for form in conjugate (verb):
        print(form)

main()


# Finding the lowest and highest letters in words

min("vary") # lowest letter in the word

max("vary") # highest letter in the word

len("vary") # length of letter

word="SHELBY" # starts indices from 0 1 2 3 4 ...

word[3] # third index letter

word[:4] # first 4 words

word[2:5] # letters from 2nd index to 5 th index

word[::2] # Picks every other word

word="palatable"

word[-1] # gives last word

word[4:] # last four words

word[1:6:2] # letters in the index 1 3 5 that is 1 through 6 with 2 increment

word[::-1] # from the back order or reverse order

word[1::2] # 2 incriment starting from 1

word[::-1]


# Determine the value of the expressions
# event in seventies
word="seventies"
word[1:6]

# seed from squelched
word="squelched"

word[0]+word[-2]+word[-2]+word[-1]

# dome in "seldom errs"
# pig in "pigeonhole"
word="pigeohole"
word[0:3]

#

phrase="string quartet in d minor"

phrase[:6]

phrase[7:12]

phrase[-5:]

phrase[2:6]

#

phrase="delivery service"

phrase[:4]

phrase[4:8]

phrase[-3:]

phrase[6::-1]

#
phrase="twenty-four hours"

phrase[:6]

phrase[7:11]

phrase[-5:-1]

phrase[3::-1]

#
phrase="salami sandwitch"

phrase[7:11]

phrase[3:6]

phrase[-3:]

phrase[3::-1]

# section 3.2
# DNA sequence
from random import choice
def complementary_base(base):
    """Return complement of single base."""
    if base == "A":
        return "T"
    elif base == "T":
        return "A"
    elif base == "C":
        return "G"
    elif base == "G":
        return "C"
    return base


def complement(dna):
    """Return complement of dna strand."""
    result = ""
    for base in dna: result += complementary_base(base)
    return result

def random_dna(length=30):
    """Return Random strand of dna of given length."""
    fragment = ""
    for _ in range(length):
        fragment +=choice ("ACGT")
    return fragment

def main():
    dna = random_dna()
    print("Sequence :", dna)
    print("Complement:", complement(dna))

main()
#----------------------------------------
# text mining from outside the book
#----------------------------------------
# NLTK-Natural Language Took Kit

import nltk
from nltk.tokenize import sent_tokenize
#this installation is must. if not installed,use nltk.download(), then select nltk.
# Q1: sentences

text="""Hello Mr. Smith, how are you doing today? The weather is great, 
            and city is awesome. The sky is pinkish-blue. You shouldn't eat cardboard"""
tokenized_text=sent_tokenize(text)
print(tokenized_text)

# Q2 words
from nltk.tokenize import word_tokenize
tokenized_word=word_tokenize(text)
print(tokenized_word)

# Q3 Frequency Distribution
from nltk.probability import FreqDist
fdist = FreqDist(tokenized_word)
print(fdist)

# Q4 most common words
fdist.most_common(2)

# Q5 Frequency distribution plot
import matplotlib.pyplot as plt
fdist.plot(30,cumulative=False)
plt.show()

import nltk
# Q6 stop words
# Stopwords considered as noise in the text.
# Text may contain stop words such as is, am, are, this, a, an, the, etc.
nltk.download ('stopwords')
nltk.download ('punkt')
from nltk.corpus import stopwords
stop_words=set(stopwords.words("english"))
print(stop_words)
len(stop_words)

# Q7 removing stop words
filtered_sent=[]
for w in tokenized_word:
    if w not in stop_words:
        filtered_sent.append(w)
        print("Tokenized Sentence:",tokenized_word)
        print("Filterd Sentence:",filtered_sent)


# Q8 stemming, keeping only root word. connection, connected, connecting -> connect

from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
ps = PorterStemmer()
stemmed_words=[]
for w in filtered_sent:
    stemmed_words.append(ps.stem(w))
print("Filtered Sentence:",filtered_sent)
print("Stemmed Sentence:",stemmed_words)

# Q9
#Lexicon Normalization
#performing stemming and Lemmatization
# Lemmatization reduces words to their base word,
# which is linguistically correct lemmas.
from nltk.stem.wordnet import WordNetLemmatizer
lem = WordNetLemmatizer()
from nltk.stem.porter import PorterStemmer

stem = PorterStemmer()
word = "flying"
print("Lemmatized Word:",lem.lemmatize(word,"v"))
print("Stemmed Word:",stem.stem(word))

# Q10 Part of speech Tagging
sent = "Albert Einstein was born in Ulm, Germany in 1879."
tokens=nltk.word_tokenize(sent)
print(tokens)
nltk.pos_tag(tokens)


# Q11 convert to lower case letter
input_str = "The 5 biggest countries by population in 2017 are China, India, United States, Indonesia, and Brazil."
input_str = input_str.lower()
print(input_str)

# Q12 removing numbers
import re
input_str = 'Box A contains 3 red and 5 white balls, while Box B contains 4 red and 2 blue balls.'
result = re.sub(r'\d+', '', input_str)
print(result)

# Q13 removing symbols: !”#$%&’()*+,-./:;<=>?@[\]^_`{|}~
import string
input_str = "This &is [an] example? {of} string. with.? punctuation!!!!" # Sample string
result = ''.join(e for e in input_str if e.isalnum())
print(result)


result = input_str.translate(str(" "," "), string.punctuation)
print(result)

# Q11 Sentimental analysis.
import pandas as pd
data=pd.read_csv('C:/Users/HOMEPCUSER/Desktop/Spring 19/PythonProgramming/train.tsv',sep='\t')
data.info()
# data has five sentiment labels
# 0-negative, 1-somewhat negative, 2-neutral, 3-somewhat positive, 4-positive
data.Sentiment.value_counts()
#test=pd.read_csv('/Users/mchowd10/Desktop/test.tsv',sep='\t')

# plot and graph
import matplotlib.pyplot as plt
Sentiment_count=data.groupby('Sentiment').count()
plt.bar(Sentiment_count.index.values, Sentiment_count['Phrase'])
plt.xlabel('Review Sentiments')
plt.ylabel('Number of Review')
plt.show()


from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
# tokenizer to remove unwanted elements from out data like symbols and numbers
# word matrix
# cv is creating document term matrix  -  matrix containing several documents as matrix.
token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(lowercase=True, stop_words='english', ngram_range = (1,1), tokenizer = token.tokenize)
text_counts= cv.fit_transform(data['Phrase'])
# Corpus: In linguistics,a corpus (plural corpora) or text corpus is a large and
# structured set of texts (nowadays usually electronically stored and processed).
# In corpus linguistics, they are used to do statistical analysis and hypothesis
# testing, checking occurrences or validating linguistic rules within a
# specific language territory.
# split data


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
text_counts, data['Sentiment'], test_size=0.3, random_state=1)
# TF-IDF: In information retrieval, tf–idf or TFIDF, short for term
# frequency–inverse document frequency, is a numerical statistic that is intended
# to reflect how important a word is to a document in a collection or corpus.
# model building and prediction
from sklearn.naive_bayes import MultinomialNB
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Generation Using Multinomial Naive Bayes
clf = MultinomialNB().fit(X_train, y_train)
predicted= clf.predict(X_test)
print("MultinomialNB Accuracy:",metrics.accuracy_score(y_test, predicted))
# IDF(Inverse Document Frequency) measures the amount of information a given
# word provides across the document. IDF is the logarithmically scaled inverse ratio
# of the number of documents that contain the word and the total number of documents.
# further modelling

from sklearn.feature_extraction.text import TfidfVectorizer
tf=TfidfVectorizer()
text_tf= tf.fit_transform(data['Phrase'])

# next splitting
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
text_tf, data['Sentiment'], test_size=0.3, random_state=123)

# model checking
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
# Model Generation Using Multinomial Naive Bayes
clf = MultinomialNB().fit(X_train, y_train)
predicted= clf.predict(X_test)
print("MultinomialNB Accuracy:",metrics.accuracy_score(y_test, predicted))



#

df=pd.read_csv('C:/Users/HOMEPCUSER/Desktop/Spring 19/PythonProgramming/kaggle.csv',index_col=0)
df.head()
df.tail()
print("There are {} observations and {} features in this dataset. \n".format(df.shape[0],df.shape[1]))
print("There are {} types of wine in this dataset such as {}... \n".format(len(df.variety.unique()),", ".join(df.variety.unique()[0:5])))
print("There are {} countries producing wine in this dataset such as {}... \n".format(len(df.country.unique()),", ".join(df.country.unique()[0:5])))

df[["country", "description","points"]].head()

# Groupby by country
country = df.groupby("country")

# Summary statistic of all countries
country.describe().head()

# top five country
country.mean().sort_values(by="points",ascending=False).head()

# some plot
plt.figure(figsize=(15,10))
country.size().sort_values(ascending=False).plot.bar()
plt.xticks(rotation=50)
plt.xlabel("Country of Origin")
plt.ylabel("Number of Wines")
plt.show()

# highest rated wine
plt.figure(figsize=(15,10))
country.max().sort_values(by="points",ascending=False)["points"].plot.bar()
plt.xticks(rotation=50)
plt.xlabel("Country of Origin")
plt.ylabel("Highest point of Wines")
plt.show()

# 1 definite question from word cloud

# word cloud
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
# Start with one review:
text = df.description[0]
# Create and generate a word cloud image:
wordcloud = WordCloud().generate(text)
# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# more word cloud
# lower max_font_size, change the maximum number of word and lighten the background:
wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# Save the image in the img folder:
wordcloud.to_file("first_review.png")
#
text = " ".join(review for review in df.description)
print ("There are {} words in the combination of all review.".format(len(text)))
# Create stopword list:
stopwords = set(STOPWORDS)
stopwords.update(["drink", "now", "wine", "flavor", "flavors"])
# Generate a word cloud image
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
# Display the generated image:
# the matplotlib way:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


###############################################################################################
################### MATH IN PYTHON ############################################################
###############################################################################################
import numpy as np
from math import e
import random
import matplotlib as mpl
import matplotlib.pyplot as plt

# Q1
print("Total:")
i = np.linspace(1,100,100)
sum(i ** 3 + 4 * i ** 2)

# Q2
print("Total:")
sum((2 ** i) / i + (3 ** i) / (i **2))

# exponents are always multiplied first, then multiplication and division, then addition and subtraction
# (order of operations), but the above code helps with readability

# Q3
print("Total:")
sum(np.exp (-i + 1) / (i + 10))

# Q4
random.seed(110) # So the same output will result each time the code is ran:
x = np.array([random.randint(0,999) for i in range(100)])
sum(np.exp (-x + 1) / (x + 10))

# Q5
def f(x_i):
    if x_i == 0:
        return 1
    else:
    # print([str(j)+'/' + str(j+1) for j in range(2,x_i+1,2)]) # for diagnostics: shows exactly what is multiplied together
    # in each iteration
        f_xi = np.prod([(ji) / (ji+1) for ji in range(2,x_i+1,2)]) #np.prod == take the product of all the listed elements
        return f_xi


series = [f(x_i) for x_i in range(0,39,2)] # 1 through 38 by 2.
sum(series)

# Q6
i = np.linspace(1,20,20).astype(int)
j = np.linspace(1,5,5).astype(int)

#same result:
print(sum([i ** 4 / (3 + j) for i in range(1,21) for j in range(1,6)]))
total = 0

for i_i in i:
    for j_i in j:
        total += (i_i ** 4) / (3 + j_i)
total


# Q7
i = np.linspace(1,20,20).astype(int)
j = np.linspace(1,5,5).astype(int)

total = []

for i_i in list(i):
    for j_i in list(j):
        total.append((i_i ** 4) / (3 + (i_i*j_i)))

sum(total)

# Q8
total = 0
iterations=[]
for i in range(1,11):
    current_iteration = []
    for j in range(1,i+1):
        total += (i ** 4) / (3 + (i*j))
        current_iteration.append(j)
    iterations.append(current_iteration)
total

import pandas as pd
# same as:
sum([(i ** 4) / (3 + (i*j)) for i in range(1,11) for j in range(1,i+1)])
print("Total:",total)
print("iterations:")
print(pd.Series(iterations))

# Q9
x = np.arange(0,1001)
total = sum(e ** -x)
print("Total:",total)

# Q10
def foo1(x,n):
    assert n > 0, "This function does not accept a number n <= 0"
    assert isinstance(n,int), "This function does not accept non-integer values"
    if n == 1:
        return 1
    result = [1] + [x**(i)/(i) for i in range(1,n)]
    return sum(result)

total = foo1(3,3)
print("Total:",total)

# Q11
x = np.linspace(-2.99,2.99,1000) # make 1000 evenly spaced values from -2.99 to 2.99
def tmpFn(xi):
    if xi < 0:
        return xi ** 2 + 2 * xi + 3

    elif xi >= 0 and xi < 2:
        return xi + 3

    elif xi >= 2:
        return xi**2+4*xi-7

tmpFn = np.vectorize(tmpFn) # when you input a list or array, the new function will apply the function to each element within it

# similar to how a for loop works
fx = tmpFn(x)
mpl.style.use("seaborn")
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


# Q12
A = np.array([[7,5,4],
[9,8,5],
[2,7,4]])

B = np.array([[9,3,6],
[7,5,5],
[1,4,5]])

print("A + B:")
print(A + B)
print()
print("A - B:")
print(A - B)
print()
print("A * B:")
print(A.dot(B))
print()
print("Determinant of A:")
print(round(np.linalg.det(A),4))
print()
print("Inverse of A:")
print(np.linalg.inv(A))
print()
print("Inverse of B:")
print(np.linalg.inv(B))
print()
print("Matrix Rank of A:")
print(np.linalg.matrix_rank(A))

# Q13
P = np.array([[2,3],
[5,7],
[4,9]])

Q = np.array([[5,6,8],
[7,4,1]])
print("P * Q")

P.dot(Q)

# Q14
print("9P")
P * 9

# Q15
I = np.array([[1,0,0],
[0,1,0],
[0,0,1]])
print('Identity Matrix:')
print(I)

# Q16
zero_matrix = np.array([0]*9).reshape(3,3)
print("Zero Matrix:")
print(zero_matrix)

# Q17
import sympy as sym
solution = sym.solve('3*x + 5','x')
solution[0]
solution = sym.solve('x**2 - 7 * x + 1','x')
solution[0]

# Q18
from sympy import *

# problem a
x, y = sym.symbols(('x','y'))
eq_system=sym.Matrix([[3,4,5],
[7,-3,7]])
solution = linsolve(eq_system,(x,y))
x,y = next(iter(solution))
print("Solution:")
print("x =", x)
print("y =", y)

# problem b
x, y, z = sym.symbols(('x','y','z'))
eq_system=sym.Matrix([[7,9,6,87],
[8,3,7,77],
[9,8,-7,13]])
solution = linsolve(eq_system,(x,y,z))
x,y,z = next(iter(solution))
print("Solution:")
print("x =", x)
print("y =", y)
print("z =", z)


##########################################################################################
###############################EXAM 01 CODE ##############################################

# Exam I

# Libraries to be imported
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt #import plotting library
import statistics as ss


# Q1
# reading csv data
d2 = pd.read_csv('C:/Users/HOMEPCUSER/Desktop/Spring 19/PythonProgramming/py1.csv')
print(d2)

# diamention of data
np.shape(d2)

# Types of data
d2.dtypes

d2.info()

# Q2
d2.isnull().sum()
# Therefore, no missing values

y1 = d2[(d2.Class == "comedy")]["Time"]
y2 = d2[(d2.Class == "drama")]["Time"]
y3 = d2[(d2.Class == "thriller")]["Time"]
y4 = d2[(d2.Class == "family")]["Time"]
y5 = d2[(d2.Class == "action")]["Time"]
y6 = d2[(d2.Class == "scifi")]["Time"]
y7 = d2[(d2.Class == "animated")]["Time"]

plt.boxplot(y1)
plt.title('Box Plot of Time for Comedy')

plt.boxplot(y2)
plt.title('Box Plot of Time for drama')

plt.boxplot(y3)
plt.title('Box Plot of Time for thriller')

plt.boxplot(y4)
plt.title('Box Plot of Time for family')

plt.boxplot(y5)
plt.title('Box Plot of Time for action')

plt.boxplot(y6)
plt.title('Box Plot of Time for scifi')

plt.boxplot(y7)
plt.title('Box Plot of Time for animated')

# Q3
print(d2.columns)
cont_var_list = ["Time","WeeklyIncome","GrossIncome","SecondViews","FirstViews"]
d3=d2[cont_var_list]
d3.describe() # summary statistics for all continuous variables

# Maximum value of time variable for each category of class variable
k = d2.groupby("Class")
print("Maximum values of Time for each category")
k.max()[["Time"]]


# Q4
datalist = [] # note that this is a list and not a dataframe


# subset the data by getting columns you need from original data d2
d9 = d2[["WeeklyIncome", "GrossIncome", "Category"]]
ds = [rows for _, rows in d9.groupby('Category')]

# I just wanted to see my list nicely to analyze the list. You could just use ds, and it will do it.
from pprint import pprint
pprint(ds)
ds[3] # good thing is you could get each dataframe by calling the appropriate number of list


len(ds)

for i in range(len(ds)):
       datalist.append(pd.DataFrame({
      'Mean.WeeklyIncome': [ss.mean(ds[i].WeeklyIncome)],
      'Mean.GrossIncome': [ss.mean(ds[i].GrossIncome)],
      'Median.WeeklyIncome': [ss.median(ds[i].WeeklyIncome)],
      'Median.GrossIncome': [ss.median(ds[i].GrossIncome)],
      'Minimum.WeeklyIncome': [min(ds[i].WeeklyIncome)],
      'Minimum.GrossIncome': [min(ds[i].GrossIncome)],
      'Maximum.WeeklyIncome': [max(ds[i].WeeklyIncome)],
      'Maximum.GrossIncome': [max(ds[i].GrossIncome)],
      'Standard_dev.WeeklyIncome': [ss.stdev(ds[i].WeeklyIncome)],
      'Standard_dev.GrossIncome': [ss.stdev(ds[i].GrossIncome)]}))

print("Please Ignore the zeroes")
pd.concat(datalist)
pprint(datalist)



# Q5
# Class
cols=['r','g','b','c','m','y','k'] #assign colors
labels=['action','animated','comedy','drama','family','scifi','thriller'] #assign labels
sizes=[49,13,50,49,9,16,31] # get this from above freqency table

# table command for frequency table
pd.crosstab(index=d2["Class"], columns="count")

# for pie chart
plt.pie(sizes,explode=None, labels=labels,colors=cols)
plt.title("Pie Chart of Class Variable")
plt.axis('equal')

# for Bar plot
x = labels
y = sizes
plt.bar(x,y)
plt.title("Bar Chart of Class Variable")
plt.xlabel("Class")
plt.ylabel("Frequency")

# Category

# table command for frequency table
pd.crosstab(index=d2["Category"], columns="count")

cols55=['r','g','b','c','m','y','k'] #assign colors
labels55=['2','3','4','5','6','7','8'] #assign labels
sizes55=[10,18,44,22,71,35,17] # get this from above freqency table

# for pie chart
plt.pie(sizes55,explode=None, labels=labels55,colors=cols55)
plt.title("Pie Chart of Category Variable")
plt.axis('equal')

# for Bar plot
x91=labels55
y91=sizes55
plt.bar(x91,y91)
plt.title("Bar Chart of Category Variable")
plt.xlabel("Types of Categories")
plt.ylabel("Frequency")



# Q7
# Creating categorical variable from numerical time
# First, you need to create an upper limit
time_limit = d2['Time'].max()+1
time_limit

# Because we will be binning the data using exclusive right binning functions,
# We make sure that the highest value is still included in the new categorical
# Label when creating bins
time_ranges = [0,120,165,time_limit] # The bins will be as follows: [0,120),[120,`165), [165, infinity)
# the labels of the new categprical variable
time_labels = [3,2,1]
# This function creates categorical variables out of quantitative
d2['Admired'] = pd.cut(d2['Time'].values, bins = time_ranges, right = False, labels = time_labels, include_lowest = False )
d2.head()

# Q8

cont_var_list = ["Time","WeeklyIncome","GrossIncome"]
d9 = d3[cont_var_list ]
P2 = d9[0 :10]
P2
AG = ss.mean(P2.mean(axis=0))
MG = max(P2.max(axis = 0 ))
mG = min(P2.min(axis=0))

d41 = P2.as_matrix()
for i in range(10) :
    for j in range(3) :
        d41[i,j] = np.power((d41[i, j] - mG/MG),AG)

print(d41)


# Q9
plt.subplot(221)
y = d2["WeeklyIncome"]
plt.hist(y)
plt.title("Histogram of WeeklyIncome")

plt.subplot(222)
y = d2["GrossIncome"]
plt.hist(y)
plt.title("Histogram of GrossIncome")

plt.subplot(223)
y = d2["SecondViews"]
plt.hist(y)
plt.title("Histogram of SecondViews")

plt.subplot(224)
y = d2["FirstViews"]
plt.hist(y)
plt.title("Histogram of FirstViews")
plt.tight_layout()



# Q6
# stacked bar chart - one above other

# Frequncy table for Class stratified by Category
pd.crosstab(index=d2["Class"], columns=d2["Category"])

# From the above frequency table you get the following data
acn = [1,5,9,10,16,5,3]
ani = [0,0,0,0,4,8,1]
com = [6,8,12,3,18,3,0]
dra = [1,2,6,5,21,7,7]
fam = [0,0,3,2,4,0,0]
scf = [0,2,5,0,3,2,4]
thr = [2,1,9,2,5,10,2]
X = range(7) # get the range of class

# Behind bar position
t1 = [1,5,9,10,20,13,4]
t2 = [7,13,21,13,38,16,4]
t3 = [8,15,27,18,59,23,11]
t4 = [8,15,30,20,63,23,11 ]
t5 = [8,17,35,20,66,25,15]
r  = [0,1,2,3,4,5,6] # bin position
names = ['Action', 'Animation','Comedy','Drama','Family','scifi','Thriller']
barWidth = 1

plt.bar(X, acn, color = 'orange', width=barWidth,label="Action" )
plt.bar(X, ani, color = '#557f2d', bottom = acn, width=barWidth, label="Animation")
plt.bar(X, com, bottom = t1, color='yellowgreen', edgecolor='white', width=barWidth, label="Comedy")
plt.bar(X, dra, bottom = t2, color='palegoldenrod', edgecolor='white', width=barWidth, label="Drama")
plt.bar(X, fam, bottom = t3, color='goldenrod', edgecolor='white', width=barWidth, label="Family")
plt.bar(X, scf, bottom = t4, color='chocolate', edgecolor='white', width=barWidth, label="scifi")
plt.bar(X, thr, bottom = t5, color='darkgoldenrod', edgecolor='white', width=barWidth, label="Thriller")
plt.title("Stack Bar Chart Class and Category",fontweight='bold')
plt.xticks(r, names, fontweight='bold')
plt.xlabel("group")
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.legend(bbox_to_anchor=(.28,.98), loc=1)

# Group Bar Chart
# set width of bar
barWidth = 0.1
r = [0.25,1.2,2.5,3.5,4.2,5.3,6.5]
# Set position of bar on X axis
r1 = np.arange(len(acn))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]
r5 = [x + barWidth for x in r4]
r6 = [x + barWidth for x in r5]
r7 = [x + barWidth for x in r6]


# Make the plot
plt.bar(r1, acn, color = 'orange', width=barWidth, edgecolor='white',label="Action" )
plt.bar(r2, ani, color = '#557f2d', edgecolor='white', width=barWidth, label="Animation")
plt.bar(r3, com, color='yellowgreen', edgecolor='white', width=barWidth, label="Comedy")
plt.bar(r4, dra, color='palegoldenrod', edgecolor='white', width=barWidth, label="Drama")
plt.bar(r5, fam, color='goldenrod', edgecolor='white', width=barWidth, label="Family")
plt.bar(r6, scf, color='chocolate', edgecolor='white', width=barWidth, label="scifi")
plt.bar(r7, thr, color='darkgoldenrod', edgecolor='white', width=barWidth, label="Thriller")
plt.title("Group Bar Chart Class and Category",fontweight='bold')
plt.xticks(r, names, fontweight='bold')
plt.xlabel("group")
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.legend(bbox_to_anchor=(.28,.98), loc=1)
plt.show()



# Q10

plt.figure(figsize=(22,18))

# Plots from  Question 2
plt.subplot(5,5,1)
plt.boxplot(y1)
plt.title('Box Plot of Time for Comedy')
plt.xticks([])              # set no ticks on x-axis
plt.yticks([])

plt.subplot(5,5,2)
plt.boxplot(y2)
plt.title('Box Plot of Time for drama')
plt.xticks([])
plt.yticks([])

plt.subplot(5,5,3)
plt.boxplot(y3)
plt.title('Box Plot of Time for thriller')
plt.xticks([])
plt.yticks([])

plt.subplot(5,5,4)
plt.boxplot(y4)
plt.title('Box Plot of Time for family')
plt.xticks([])
plt.yticks([])


plt.subplot(5,5,5)
plt.boxplot(y5)
plt.title('Box Plot of Time for action')
plt.xticks([])
plt.yticks([])

plt.subplot(5,5,6)
plt.boxplot(y6)
plt.title('Box Plot of Time for scifi')
plt.xticks([])
plt.yticks([])

plt.subplot(5,5,7)
plt.boxplot(y7)
plt.title('Box Plot of Time for animated')
plt.xticks([])
plt.yticks([])

plt.subplot(5,5,8)
# Plots from Question 5
# for Class -  Pie chart
cols1=['r','g','b','c','m','y','k'] #assign colors
sizes1=[49,13,50,49,9,16,31] # get this from above freqency table
plt.pie(sizes1,explode=None, colors=cols1)
plt.title("Pie Chart of Class Variable")
plt.axis('equal')



plt.subplot(5,5,9)
# Bar plot for class
labels1 = ['ac','an','co','dr','fm','sc','th'] #assign labels
sizes1 = [49,13,50,49,9,16,31]
x11 = labels1
y11 = sizes1
plt.bar(x11,y11)
plt.title("Bar Chart of Class Variable")



plt.subplot(5,5,10)
# for Category -  Pie chart
plt.pie(sizes55,explode=None, colors=cols55)
plt.title("Pie Chart of Category Variable")
plt.axis('equal')

plt.subplot(4,5,11)
# Bar plot for Category
labels2=['2','3','4','5','6','7','8']
x22 = labels2
y22 = sizes55
plt.bar(x22,y22)
plt.title("Bar Chart of Category Variable")



# Plots for Question 9
plt.subplot(5,5,12)
y41 = d2["WeeklyIncome"]
plt.hist(y41)
plt.title("Histogram of WeeklyIncome")

plt.subplot(5,5,13)
y42 = d2["GrossIncome"]
plt.hist(y42)
plt.title("Histogram of GrossIncome")

plt.subplot(5,5,14)
y43 = d2["SecondViews"]
plt.hist(y43)
plt.title("Histogram of SecondViews")

plt.subplot(5,5,15)
y44 = d2["FirstViews"]
plt.hist(y44)
plt.title("Histogram of FirstViews")



# Plot from Question 6
plt.subplot(5,5,17)
# Group Bar Chart
# set width of bar
barWidth = 0.1
# Set position of bar on X axis
r1 = np.arange(len(acn))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]
r5 = [x + barWidth for x in r4]
r6 = [x + barWidth for x in r5]
r7 = [x + barWidth for x in r6]


# Make the plot
plt.bar(r1, acn, color = 'orange', width=barWidth, edgecolor='white',label="Action" )
plt.bar(r2, ani, color = '#557f2d', edgecolor='white', width=barWidth, label="Animation")
plt.bar(r3, com, color='yellowgreen', edgecolor='white', width=barWidth, label="Comedy")
plt.bar(r4, dra, color='palegoldenrod', edgecolor='white', width=barWidth, label="Drama")
plt.bar(r5, fam, color='goldenrod', edgecolor='white', width=barWidth, label="Family")
plt.bar(r6, scf, color='chocolate', edgecolor='white', width=barWidth, label="scifi")
plt.bar(r7, thr, color='darkgoldenrod', edgecolor='white', width=barWidth, label="Thriller")
plt.title("Group Bar Chart Class and Category")
plt.xlabel("group")
plt.xlabel("Class")
plt.ylabel("Frequency")


plt.subplot(5,5,19)
# Plot from Question 6

# stacked bar chart - one above other
acn = [1,5,9,10,16,5,3]
ani = [0,0,0,0,4,8,1]
com = [6,8,12,3,18,3,0]
dra = [1,2,6,5,21,7,7]
fam = [0,0,3,2,4,0,0]
scf = [0,2,5,0,3,2,4]
thr = [2,1,9,2,5,10,2]
X = range(7) # get the range of class

# Behind bar position
t1 = [1,5,9,10,20,13,4]
t2 = [7,13,21,13,38,16,4]
t3 = [8,15,27,18,59,23,11]
t4 = [8,15,30,20,63,23,11 ]
t5 = [8,17,35,20,66,25,15]
r  = [0,1,2,3,4,5,6] # bin position
names = ['Action', 'Animation','Comedy','Drama','Family','scifi','Thriller']
barWidth = 1
plt.bar(X, acn, color = 'orange', width=barWidth,label="Action" )
plt.bar(X, ani, color = '#557f2d', bottom = acn, width=barWidth, label="Animation")
plt.bar(X, com, bottom = t1, color='yellowgreen', edgecolor='white', width=barWidth, label="Comedy")
plt.bar(X, dra, bottom = t2, color='palegoldenrod', edgecolor='white', width=barWidth, label="Drama")
plt.bar(X, fam, bottom = t3, color='goldenrod', edgecolor='white', width=barWidth, label="Family")
plt.bar(X, scf, bottom = t4, color='chocolate', edgecolor='white', width=barWidth, label="scifi")
plt.bar(X, thr, bottom = t5, color='darkgoldenrod', edgecolor='white', width=barWidth, label="Thriller")
plt.title("Stack Bar Chart Class and Category")
plt.xlabel("group")
plt.xlabel("Class")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()


#####################################################################################
###########################EXAM 02 CODE##############################################

import black
import scipy.stats as stats
import math
import scipy
import pandas as pd
import numpy as np
import scipy.stats as stats


n1 = 1000 # size of dataset

# X coming from normal distribution
mu = 5  # mean
sd1 = 1.5  # scale
X = np.random.normal(mu, sd1, n1)  # mean, variance, size
print(X)

# U coming from uniform distribution
U = np.random.uniform(1, 6, n1).round(2)
print(U)


# Epsilon coming from exponential distribution
lam = 1/3  # lambda value
epsilon = np.random.exponential(lam, n1).round(2)
print(epsilon)



# Create function
def f(X, U, epsilon):
    y = 100 + (3.5 * np.exp(X) - math.pi * np.log(300+U) + epsilon
    return y

# Now that we have created values, let us call the function
y1 = f(X, U, epsilon)

# we have obtained y1 as array.
# Next we compute mean and variance of y1
np.mean(y1)

np.var(y1)

# test whether mean is 1000 or not
scipy.stats.ttest_1samp(y1, 1000)


# converting y1 to dataframe
yd = pd.DataFrame(data=y1)
np.mean(yd)

# test whether variance is 5000000 or not
def chi_sq_test_for_variance(variable, h0) :  # function for the chi squared distribution associated with the above formula

    sample_variance = variable.var()  # Find the variance of the sample
    n = variable.notnull().sum()  # Take the sum of the number of values that are not missing
    # the actual number of observations for the variable where
    # True = 1, False = 0
    degrees_of_freedom = n - 1  # Find the degrees of freedom
    x_sq_stat = (n - 1) * sample_variance / h0  # Using the formula above to calculate the X^2 statistic
    p = stats.chi2.cdf(x_sq_stat, degrees_of_freedom)  # Here, a cumulative distribution function is used to determine
    # the significance of the variance using the X^2 statistic.

    # If a chi square test statistic is over the 99th percentile,
    # we'd have reason to suspect significance at alpha = .05.
    # We need to account for circumstance where the p value is greater
    # than .05, however:
    if p > .5 :
        p = 1 - p
    return (x_sq_stat, p, degrees_of_freedom)  # End of function

y1_variance = round(yd[0].var(), 2)
x_sq_stat, pval, dof = chi_sq_test_for_variance(yd[0], h0=5000000)
print('Chi-square statistic, pvalue, degrees of freedom')
print(x_sq_stat, pval, dof )

# Compute confidence interval for mean
import statsmodels.stats.api as sms
sms.DescrStatsW(yd[0]).tconfint_mean()  # Confidence interval of mean for y1 -  default confidence level is 95%


# Next we repeat simulation 100 times
def simdata(n):
    j = [i for i in range(1, n+1)]
    y00 = np.random.normal(mu, sd1, n)
    y11 = np.random.uniform(1, 6, n).round(2)
    y12 = np.random.exponential(lam, n).round(2)
    y13 = f(y00,y11,y12)
    y14 = y13.mean()
    m1 = pd.DataFrame({'sample_size': j,
                       'normal': y00,
                       'uniform': y11,
                       'exponential': y12,
                       'y value': y13,
                        'mean value': y14 })

    ds = [rows for _, rows in m1.groupby('sample_size')]
    return (m1,ds)


# simulate 100 times randomly
simulate, ds = simdata(100)

print(simulate['y value'])

# Next we compute mean and variance of simulated y value
np.mean(simulate['y value'])
np.mean(simulate['mean value'])
np.var(simulate['y value'])

# Now take a look at simulated data
print(simulate)



