# Exam I

# Libraries to be imported
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt #import plotting library
import statistics as ss


# Q1
# reading csv data
d2=pd.read_csv('C:/Users/HOMEPCUSER/Desktop/Spring 19/PythonProgramming/py1.csv')
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
d9 = d2[["WeeklyIncome","GrossIncome","Category"]]
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
