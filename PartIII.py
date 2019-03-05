

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
# Test the equality of proportion for smoking when RACE=1 and RACE=2.
# A hypothesis test formally tests if the proportion of smoking in RACE1 and RACE2 populations are equal.

# When one variable is an explanatory variable (X, fixed) and the other a response variable (Y, random),
# the hypothesis of interest is whether the populations have the same or different proportions in each category.

a = pd.crosstab(d9[d9.RACE == 1].SMOKE, columns="counts")
b = pd.crosstab(d9[d9.RACE == 2].SMOKE, columns="counts")
print(a)
print(b)

# add value like a[1], a[0], b[1], b[0]
aa = np.array([[918, 51], [950, 23]])
print(stats.chi2_contingency(aa))  # chi-squared value and p value

# Test method. Use the chi-square goodness of fit test to determine whether observed sample frequencies
# differ significantly from expected frequencies specified in the null hypothesis.

# Q5
# Construct a confidence interval of mean for SBP and TC.

import statsmodels.stats.api as sms
sms.DescrStatsW(d9.AGE).tconfint_mean()  # Confidence interval of mean for AGE -  default confidence level is 95%
sms.DescrStatsW(d9.SBP).tconfint_mean()  # Confidence interval of mean for SBP -  default confidence level is 95%
sms.DescrStatsW(d9.TC).tconfint_mean()  # Confidence interval of mean for TC -  default confidence level is 95%


# alternative approach for the same problem -  you could use this for different confidence level
stats.t.interval(0.95, len(d9.AGE)-1, loc=np.mean(d9.AGE), scale=stats.sem(d9.AGE))
stats.t.interval(0.95, len(d9.SBP)-1, loc=np.mean(d9.SBP), scale=stats.sem(d9.SBP))
stats.t.interval(0.95, len(d9.TC)-1, loc=np.mean(d9.TC), scale=stats.sem(d9.TC))

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
# Construct a confidence interval for proportion of smokers.

table = pd.crosstab(d9['SMOKE'], columns='count')
print(table)

import statsmodels.stats.proportion as one
ci_low, ci_upp = one.proportion_confint(74, 1868, alpha=0.05, method='normal')
print(ci_low, ci_upp)

# Q8
# Also construct a confidence interval for difference of proportions for smokers when RACE=1 and RACE=2.

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
# Non- parametric test of median
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

a1=pd.crosstab(d9.CSBP, columns="counts")
print(a1)
b1=pd.crosstab(d9.CWT, columns="counts")
print(b1)
# get value from a1 and b1
bb=np.array([[17,1650,275],[480,705,757]])
scipy.stats.chi2_contingency(bb)

def f(x):
    y = math.pow(x, 2)+math.exp(-x)+math.log(5*x)+2
    return y
