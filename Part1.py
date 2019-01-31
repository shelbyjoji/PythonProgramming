
def main():
    print(myfirstvariable)

def helloworld(helloworld):
    print(helloworld)

myfirstvariable = "Hello World!!"
main()
print(myfirstvariable)
helloworld(myfirstvariable)



# Part I

import pandas as pd
import numpy as np

#Q1
# reading csv data
d1=pd.read_csv('C:/Users/HOMEPCUSER/Desktop/Spring 19/PythonProgramming/PROJECTDATA.csv')
print(d1)

# reading text data
text= pd.read_table('C:/Users/HOMEPCUSER/Desktop/Spring 19/PythonProgramming/ProjectData.txt')
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
d1.describe()[['SBP','DBP']]

#subset the data and then describe
d3 = d1[['SBP','DBP']]
d3.describe()

#Q7
#create column for missing values in each row - axis = 0 for column, axis = 1 for rows
d1["MISSING"] = d1.isnull().sum(axis=1)

#Q8 & #Q9 & #Q10
#Here we drop rows that has missing value.
#Create a new data d2 by deleting any observation (any row) that has missing value
d2=d1.dropna()
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


#Q12
#create ABP (Average Blood Pressure) variable by averaging SBP and DBP for each row of th
initial_bp_variables = ['SBP', 'DBP']
#Remember axis =1 gives you rows
#mean(axis=1) gives average of selected rows
d2["ABP"] = d2[initial_bp_variables].mean(axis=1)
d2.tail()


d4=d2[["RACE","SBP"]]
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
#a frequency table for the BPC variable -  frequncy table for a categorical variable
pd.crosstab(index=d2["BPC"],columns="Count")
pd.crosstab(index=d2["SMOKE"],columns="Count")

# Q15
#Frequncy table for BPC stratified by race
pd.crosstab(index=d2["BPC"],columns=d2["RACE"])
#ABP is for average BP. It does not make any sense to use ABP for frequncy table as it is quantitative

# Q17
#Look smoke is categorical
#find variance of TC for SMOKE variable
bysmoke = d2.groupby('SMOKE')
bysmoke.mean()
bysmoke.std()#to get variance
K=bysmoke.var()#to get variance
K[["TC"]]

# Q16
#FInd mean of TC for RACE variable
bysmoke = d2.groupby('RACE')
k= bysmoke.mean()
bysmoke.std()
k[["TC"]]


# Q18
#In this section, we draw bar chart and pie chart
import matplotlib.pyplot as plt #import plotting library

cols=['r','g'] #assign colors
labels=['NonSmoker','Smoker'] #assign labels


# table command for frequency table
pd.crosstab(index=d2["SMOKE"], columns="count")
sizes=[74,1868] # get this from above freqency table
explode = [0,0.1] # this will make the piece of pie protrude out; None will keep it within the circle
#for pie chart
plt.pie(sizes,explode=explode, labels=labels,colors=cols)
#for axis and titles
plt.title("Pie Chart of Race Variable")
plt.axis('equal')

# bar chart
x=['NonSmoker','Smoker'] # name categories
y=sizes
x
y
plt.bar(x,y)
plt.title("Bar Chart of Smoke Variable")
plt.xlabel("Smoking Status")
plt.ylabel("Frequency")

# you could do the same for RACE and INCOME

# Q19
#Frequncy table for RACE stratified by INCOME
pd.crosstab(index=d2["RACE"], columns=d2["INCOME"])

#Frequncy table for INCOME stratified by RACE
pd.crosstab(index=d2["INCOME"], columns=d2["RACE"])


#stacked bar chart
#One above other
import matplotlib.pyplot as plt

#Frequncy table for RACE stratified by INCOME
pd.crosstab(index=d2["RACE"], columns=d2["INCOME"])
#From the above frequency table you get A (for RACE 1) and B(for RACE 2)
A = [14,18,13,80,149,166,167,249,113]
B = [87,71,57,188,164,143,86,155,22]
X = range(9) # get the range of income
X
plt.bar(X, A, color = 'b',label="Race1" )
plt.bar(X, B, color = 'r', bottom = A, label="Race2")

plt.title("Stack Bar Chart Income and Race")
plt.xlabel("Income Status")
plt.ylabel("Frequency")
plt.legend(bbox_to_anchor=(.22,.98), loc=1)

#Q20
#Now replot the same as unstacked
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
plt.xticks([r + barWidth for r in range(len(A))], [1,2,3,4,5,6,7,8,9])
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

# subsetting data
#Here you table DBP by race. Basically you are extracting it into two dataset x1 and x2
x1=d2[(d2.RACE == 1)]["DBP"]
x2=d2[(d2.RACE == 2)]["DBP"]
x1 # look how x1 is

#plotting two seperate graphs
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

# Draw histogram for SBP
y = d2["SBP"]
plt.hist(y)
plt.title("Histogram of SBP")
plt.xlabel("Values of SBP")
plt.ylabel("Frequency")

# Draw two separate histograms for SBP when RACE=1 and RAce=2
# two histogram together in one graph
y1=d2[(d2.RACE == 1)]["SBP"]
y2=d2[(d2.RACE == 2)]["SBP"]
plt.hist(y1)
plt.title('SBP for african american')
plt.xlabel("Values of SBP")
plt.hist(y2)
plt.title('SBP for caucasian')
plt.xlabel("Values of SBP")

# ------------
# Q23
# ------------
#Draw boxplot for SBP
plt.boxplot(y)
plt.title("Boxplot of SBP")

# Draw two separate boxplot for SBP when RACE=1 and RAce=2
# two box plot together. If you run it together, it will be in same graph.
plt.boxplot(y1)
plt.title('Box Plot of SBP for african american')
plt.boxplot(y2)
plt.title('Box Plot of SBP for caucasian')

# ----------------------
# Q24, all 9 graphs
# ----------------------

#Creating subplots
# here we are subplotting on 3 by 3 grid
#331,332,333
#334, 335, 336
#337, 338,339

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
#create a random of 200 observations
random_subset = d2.sample(n=50,replace=False)

# stem-leaf plot
# ------------
# Q26
# ------------
k=round(random_subset.SBP)
k
y = pd.Series(k)
plot.stem(y)

from stemgraphic import stem_graphic
fig, axes = stem_graphic(y)

# ------------
# Q27
# ------------
#subsetting data
#subset the d1 dataframe by excluding values
exclude = ["ID","TC", "TG", "HDL", "LDL"]
keep = [var for var in d1.columns if var not in exclude]
d3 = d1[keep].copy()
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
#Subset the data again by keeping age from 9 to 20
d5= d4.loc[(d4["AGE1"] >= 9) & (d4['AGE1']<=20)].copy()

#------------
# Q32
#------------
#exporting the dataframe d5 as csv format. You will find it in the same folder for python code file
d5.to_csv("d5_Cleaned_data.csv",index=False)

#------------
# Q33
#------------
d6 = d5.copy().groupby("AGE1")
d6.head()

#------------
# Q34
#------------
# get number of observations by each age category
print("Observations by Age:")
print(d6.size())

print("Total Observations:")
print(sum(d6.size()))

print("Number of age categories")
len(d6)

#print group by age category of 11
d6.get_group(11).head()

####################ERROR***DO**NOT**RUN######################
def splitframe(data, name='name') :
    n = data[name][0]
    df = pd.DataFrame(columns=data.columns)
    datalist = []
    for i in range(len(data)) :
        if data[name][i] == n :
            df = df.append(data.iloc[i])
        else :
            datalist.append(df)
            df = pd.DataFrame(columns=data.columns)
            n = data[name][i]
            df = df.append(data.iloc[i])
    return datalist
k = splitframe(d5, name="AGE1")
################################################


# ------------------
# Q35 and Q36
# ------------------
# pseudocode
# Analyze the columns you need and subset the data
# Drop missing variables and create new data frame. Missing values may interfere with the calculations
# Again subset the age group required to new dataFrame
# Now get the rounded value of age into new variable called NewAge
# Create a new list grouped by NewAge. This will create a dataframe under each list
# You will find that the length of the list will be same as number of categories
# Now iterate through each item of list, Within each dataframe, use ss library to find appropritate statistics
# Bind the list together by concatenate function.

import statistics as ss
datalist=[] # note that this is a list and not a dataframe


#subset the data by getting columns you need from original data d1
d8=d1[['RACE','INCOME','AGE','SMOKE','SBP','DBP','HT', 'WT','WM','BMI']]
d9=d8.dropna() # remove missing value rows
d11=d9[(d9.AGE > 9) & (d9.AGE <= 19)] # subset the data by age from 10 to 19
d11['NewAge']=round(d11.AGE,0)#now you round it
ds = [rows for _, rows in d11.groupby('NewAge')]

#I just wanted to see my list nicely to analyze the list. You could just use ds, and it will do it.
from pprint import pprint
pprint(ds)
ds[1] # good thing is you could get each dataframe by calling the appropriate number of list
d11.groupby("NewAge").mean() #by Mr,sami- Question is why the f**k do I need this line -  don't run it!

len(ds)
for i in range(len(ds)):
       datalist.append(pd.DataFrame({'age':[ss.mean(ds[i].NewAge)],
      'Mean.SBP': [ss.mean(ds[i].SBP)],
      'Mean.DBP': [ss.mean(ds[i].DBP)],
      'Mean.HT': [ss.mean(ds[i].HT)],
      'Mean.WT': [ss.mean(ds[i].WT)],
      'Mean.BMI': [ss.mean(ds[i].BMI)]}))

pd.concat(datalist)

pprint(datalist)
datalist[1]

# Great !!! Buhahaaha!!


# ------------
# Q37, Q38, Q39
# ------------
import statistics as ss
# Without double loop
D4 = d9[0 :10]
G = ss.mean(D4.mean(axis=0))
D4 - G
np.power((D4 - G), 2)

# double loop, How to write the double loop
d41 = D4.as_matrix()
for i in range(10) :
    for j in range(10) :
        d41[i, j] = (d41[i, j] - G) ** 2
d41

# ------------
# Q40
# ------------
D1 = pd.read_csv("PROJECTDATA.csv")
# look data is longitudinal
# you cannot do regular regression - look into mixed regression

del D1['Unnamed: 0']
D8 = D1.copy(deep=True)
D8['NEW.ID'] = D8['ID'] - 100010
del D8['ID']
D8.head()

# ------------
# Q41
# ------------
D8_column_list = D8.columns
sorted_D8_columns = sorted(D8_column_list)
D8 = D8[sorted_D8_columns]
D8.head()

# ------------
# Q42
# ------------
# wide format
D8 = D8[['NEW.ID', 'VISIT', 'SBP']]
D8['VISIT'].loc[D8['VISIT'] == 0] = 1
W1 = D8.pivot(index="NEW.ID", columns='VISIT').sort_index(axis=1, level=1).sort_index().copy(deep=True)
W1

# ------------
# Q43
# ------------
#Long format
L1 = W1.copy().stack(level=1).reset_index(level=1)
sorted_L1_columns = sorted(L1.columns)
L1 = L1[sorted_L1_columns]
L1

# ------------
# Q44
# ------------
columns = ['AGE', 'SBP', 'DBP', 'HT', 'WT', 'BMI']
column_list_length = len(columns)
D2 = D1[columns].copy().dropna()
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


print("Null Proportions:")
D2.apply(calculate_null_proportion)
# ------------
# Q45
# ------------
columns = ["ID", "AGE", "SBP", "DBP"]
N1 = D1[columns].copy()
# ------------
# Q46
# ------------
columns = ["ID", "TC", "TG", "HT", "WT"]
N2 = D1[columns].copy()
N2
# ------------
# Q47
# ------------
N_merged = N1.merge(N2, left_on='ID', right_on='ID')
N_merged.head()
# ------------
# Q48
# ------------
M1 = D1.iloc[:10000].sample(1000, replace=False, random_state=2)
M1
# ------------
# Q49
# ------------
M2 = D1.loc[~D1.index.isin(M1.index)].sample(1000, replace=False, random_state=3)
# The "~" symbol states to get everything where the following statement is NOT true (negation)
M2
# ------------
# Q50
# ------------
M_merged = M1.append(M2)
M_merged
