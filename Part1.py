
def main():
    print(myfirstvariable)

def helloworld(helloworld):
    print(helloworld)

myfirstvariable = "Hello World!!"
main()
print(myfirstvariable)
helloworld(myfirstvariable)




#Part I

import pandas as pd
import numpy as np

#reading csv data
d1=pd.read_csv('C:/Users/HOMEPCUSER/Desktop/Spring 19/PythonProgramming/PROJECTDATA.csv')
print(d1)

#reading text data
text= pd.read_table('C:/Users/HOMEPCUSER/Desktop/Spring 19/PythonProgramming/ProjectData.txt')
print(text)

# to get the dimention of dataset
d1.shape
np.shape(d1)

#count variables  that are integer and numeric
d1.info()
d1.dtypes

#number of variables
print(d1.columns)

#Delete the ID variable from the data set
del d1['ID']
del d1['Unnamed: 0']

#get the head
d1.head(10)

# get the tail
d1.tail(10)

#missing values for each variables -  it reports the null value
d1.isnull().sum()

#obtain summary statistics
d1.describe()
d1.describe()[['SBP','DBP']]

#subset the data and then describe
d3 = d1[['SBP','DBP']]
d3.describe()

#column for missing values in each row -  axis =2 for column
d1["MISSING"] = d1.isnull().sum(axis=1)

#Create a new data D2 by deleting any observation (any row) that has missing value
d2=d1.dropna()
d2.shape
d2.describe()

#Use the apply function (R code) to report standard deviation for numerical variables only
irrelevant_variables_list = ["RACE","INCOME","SMOKE","BREAST","WAIST","EVENT"]
relevant_columns = [variable for variable in d2.columns if variable not in irrelevant_variables_list]
d2[relevant_columns].std()

#Q12
initial_bp_variables = ['SBP','DBP']
d2["ABP"] = d2[initial_bp_variables].mean(axis=1)
d4=d2[["RACE","SBP"]]
d4.tail()

# subsetting data
all_quant_bp_variables = ['SBP','DBP','ABP']
d2[all_quant_bp_variables].head()
d2.head()

#Q13
abp_limit = d2['ABP'].max()+1 # Because we will be binning the data using exclusive right ginning functions,
# We make sure that the highest value is still included in the new categorical
# Label when creating bins

bpc_ranges = [0,85,100,abp_limit] # The bins will be as follows: [0,85),[85,`100), [100, infinity)
bpc_labels = [3,2,1] # the labels of the new categprical variable
# This function creates categorical variables out of quantitative
d2['BPC'] = pd.cut(d2['ABP'].values, bins = bpc_ranges, right = False, labels = bpc_labels, include_lowest = False )

# variables by "binning" them.
new_bp_variables = ['ABP','BPC']
all_bp_variables = new_bp_variables+initial_bp_variables # with lists, we can join them together using
# list1 + list2
print("Preview of All Blood Pressure Variables:")
print(d2[all_bp_variables])


#Q14
pd.crosstab(index=d2["BPC"],columns="Count")

#Q15
pd.crosstab(index=d2["BPC"],columns=d2["RACE"])

# Q16
bysmoke = d2.groupby('SMOKE')
bysmoke.mean()
bysmoke.std()

# Q17
bysmoke = d2.groupby('RACE')
bysmoke.mean()
bysmoke.std()

# Q18
import matplotlib.pyplot as plt
cols=['r','g']
labels=['NonSmoker','Smoker']
sizes=[75,1985]

# table command for frequency table
pd.crosstab(index=d2["SMOKE"], columns="count")
explode = [0,0.1]
plt.pie(sizes,explode=explode, labels=labels,colors=cols)
plt.title("Pie Chart of Race Variable")
plt.axis('equal')

# bar chart
x=['NonSmoker','Smoker']
y=[75,1985]
plt.bar(x,y)
plt.title("Bar Chart of Race Variable")
plt.xlabel("Smoking Status")
plt.ylabel("Frequency")

# Q19

pd.crosstab(index=d2["RACE"], columns=d2["INCOME"])
pd.crosstab(index=d2["INCOME"], columns=d2["RACE"])

#stacked bar chart
import matplotlib.pyplot as plt
A = [14,18,13,80,149,166,167,249,113]
B = [87,71,57,188,164,143,86,155,22]
X = range(9)
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
r2 = [x + barWidth for x in r1]
# Make the plot
plt.bar(r1, A, color='orange', width=barWidth, edgecolor='white', label='Race1')
plt.bar(r2, B, color='steelblue', width=barWidth, edgecolor='white', label='Race2')
# Add xticks on the middle of the group bars
plt.xlabel('Income Status', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(A))], [1,2,3,4,5,6,7,8,9])
# Create legend & Show graphic
plt.legend()
plt.show()
plt.title("Grouped Bar Chart for Income and Race")
plt.ylabel("Frequency")


# dot plot for DBP5. Also draw two separate dot plots for DBP5 when RACE=1 and RA
#fix labels and titles of 3 by 3 graphs
#do the stem deaf plot - sbp, dbp


#------------
# Q21
#------------
x = d2["DBP"]
plt.plot(x,'bo')
plt.title('Dotplot.DBP')
# subsetting data
x1=d2[(d2.RACE == 1)]["DBP"]
x2=d2[(d2.RACE == 2)]["DBP"]
plt.plot(x1,'bo')
plt.title('Dotplot.CC')
plt.grid(True)
plt.plot(x2,'bo')
plt.title('Dotplot.AA')
plt.grid(True)
#------------
# Q22
#------------
y = d2["SBP"]
plt.hist(y)
plt.title("Histogram of SBP")
plt.xlabel("Values of SBP")
plt.ylabel("Frequency")
# two histogram together
y1=d2[(d2.RACE == 1)]["SBP"]
y2=d2[(d2.RACE == 2)]["SBP"]
plt.hist(y1)
plt.title('SBP for african american')
plt.xlabel("Values of SBP")
plt.hist(y2)
plt.title('SBP for caucasian')
plt.xlabel("Values of SBP")
#------------
# Q23
#------------
plt.boxplot(y)
plt.title("Boxplot of SBP")
# two boxplot together
plt.boxplot(y1)
plt.title('Box Plot of SBP for african american')
plt.boxplot(y2)
plt.title('Box Plot of SBP for caucasian')
#----------------------
# Q24, all 9 graphs
#----------------------
# 1. three dot plot
plt.subplot(331)
plt.plot(x,'bo')
plt.title('Dotplot of DBP')
plt.subplot(332)
plt.plot(x1,'bo')
plt.title('Dotplot of CC')
plt.grid(True)
plt.subplot(333)
plt.plot(x2,'bo')
plt.title('Dotplot of AA')
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
# 3. Three Boxplot
plt.subplot(337)
plt.boxplot(y)
plt.title("Boxplot of SBP")
# two boxplot together
plt.subplot(338)
plt.boxplot(y1)
plt.title('Box Plot of SBP for AA')
plt.subplot(339)
plt.boxplot(y2)
plt.title('Box Plot of SBP for CC')
#------------
# Q25
#------------
import matplotlib.pyplot as plot
import pylab
import math
# method 1, by selecting random rows
random_subset = d2.sample(n=50,replace=False)
# stem-leaf plot
#------------
# Q26
#------------
y = pd.Series(round(random_subset.WT))
plot.stem(y)
#------------
# Q27
#------------
exclude = ["ID","TC", "TG", "HDL", "LDL"]
keep = [var for var in d1.columns if var not in exclude]
d3 = d1[keep].copy()
d3.head()
#------------
# Q28
#------------
np.shape(d3)
d3.isnull().sum()
d3["MISSING"] = d3.isnull().sum(axis=1)
#------------
# Q29
#------------
d4=d3.dropna()
#------------
# Q30
#------------
d4['AGE1'] = round(d4['AGE']).astype(int)
#------------
# Q31
#------------
d5=d4.loc[(d4["AGE1"] >= 9) & (d4['AGE1']<=20)].copy()
#------------
# Q32
#------------
d5.to_csv("d5_Cleaned_data.csv",index=False)
#------------
# Q33
#------------
d6 = d5.copy().groupby("AGE1")
d6.head()
#------------
# Q34
#------------
print("Observations by Age:")
print(d6.size())
print()
print("Total Observations:")
print(sum(d6.size()))
len(d6)
d6.get_group(11).head()


for AGE1, d5_AGE1 in d5.groupby('AGE1'):
    print(d5_AGE1)
