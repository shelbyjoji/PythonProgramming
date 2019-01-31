
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

# reading csv data
d1=pd.read_csv('C:/Users/HOMEPCUSER/Desktop/Spring 19/PythonProgramming/PROJECTDATA.csv')
print(d1)

# reading text data
text= pd.read_table('C:/Users/HOMEPCUSER/Desktop/Spring 19/PythonProgramming/ProjectData.txt')
print(text)

# to get the dimension of dataset
d1.shape
np.shape(d1)

# count variables  that are integer and numeric
d1.info()
d1.dtypes

# number of variables
print(d1.columns)

# Delete the ID variable from the data set
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
initial_bp_variables = ['SBP', 'DBP']
d2["ABP"] = d2[initial_bp_variables].mean(axis=1)
d4=d2[["RACE","SBP"]]
d4.tail()
d4

# subsetting data
all_quant_bp_variables = ['SBP','DBP','ABP']
d2[all_quant_bp_variables].head()
d2.head()

#Q13
abp_limit = d2['ABP'].max()+1
# Because we will be binning the data using exclusive right ginning functions,
# We make sure that the highest value is still included in the new categorical
# Label when creating bins
bpc_ranges = [0,85,100,abp_limit] # The bins will be as follows: [0,85),[85,`100), [100, infinity)
# the labels of the new categprical variable
bpc_labels = [3,2,1]
# This function creates categorical variables out of quantitative
d2['BPC'] = pd.cut(d2['ABP'].values, bins = bpc_ranges, right = False, labels = bpc_labels, include_lowest = False )
# variables by "binning" them.
new_bp_variables = ['ABP','BPC']
# with lists, we can join them together using list1 + list2
all_bp_variables = new_bp_variables+initial_bp_variables
print("Preview of All Blood Pressure Variables:")
print(d2[all_bp_variables])


# Q14
pd.crosstab(index=d2["BPC"],columns="Count")

# Q15
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
plt.title("Grouped Bar Chart for Income and Race")
plt.ylabel("Frequency")
plt.show()

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

plt.show()

# ------------
# Q22
# ------------
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

# ------------
# Q23
# ------------
plt.boxplot(y)
plt.title("Boxplot of SBP")
# two box plot together
plt.boxplot(y1)
plt.title('Box Plot of SBP for african american')
plt.boxplot(y2)
plt.title('Box Plot of SBP for caucasian')

# ----------------------
# Q24, all 9 graphs
# ----------------------
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
random_subset = d2.sample(n=50,replace=False)

# stem-leaf plot
# ------------
# Q26
# ------------
k=round(random_subset.WT)
y = pd.Series(k)
plot.stem(y)

from stemgraphic import stem_graphic
fig, axes = stem_graphic(y)

# ------------
# Q27
# ------------
exclude = ["ID","TC", "TG", "HDL", "LDL"]
keep = [var for var in d1.columns if var not in exclude]
d3 = d1[keep].copy()
d3.head()

# ------------
# Q28
# ------------
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

##########################################
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

d8=d1[['RACE','INCOME','AGE','SMOKE','SBP','DBP','HT', 'WT','WM','BMI']]
d9=d8.dropna()
d11=d9[(d9.AGE > 9) & (d9.AGE <= 19)]
d11['NewAge']=round(d11.AGE,1)
ds = [rows for _, rows in d11.groupby('NewAge')]
d11.groupby("NewAge").mean() # mr. Sami
len(ds)
import statistics as ss
datalist=[]
for i in range(len(ds)):
       datalist.append(pd.DataFrame({'age':[ss.mean(ds[i].NewAge)],'Mean.SBP': [ss.mean(ds[i].SBP)],
      'Mean.DBP': [ss.mean(ds[i].DBP)],
      'Mean.HT': [ss.mean(ds[i].HT)],
      'Mean.WT': [ss.mean(ds[i].WT)],
      'Mean.BMI': [ss.mean(ds[i].BMI)]}))
pd.concat(datalist)




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
