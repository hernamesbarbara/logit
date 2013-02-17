import numpy as np
import pandas as pd


##################################################
#### Description of the data
##################################################
f = "http://www.ats.ucla.edu/stat/data/binary.csv"
df = pd.read_csv(f)

## view the first few rows of the data
df.head()
"""
   admit  gre   gpa  rank
0      0  380  3.61     3
1      1  660  3.67     3
2      1  800  4.00     1
3      1  640  3.19     4
4      0  520  2.93     4
"""

predictors = ['gre', 'gpa', 'rank']
target	   = 'admit'

# Continuous = ['gre', 'gpa']
# Discrete   = ['rank']
ranks = sorted(df['rank'].unique())
ranks = pd.Series([i for i in reversed(ranks)])
ranks = pd.Factor(labels=ranks.index, levels=ranks.values, name="ranks")
df['rank'] = df['rank'].apply(lambda x: ranks.levels.get_loc(x))
##################################################

##################################################
#### Summarizing the data
df.describe()      # akin to summary(df) in R
df.apply(np.std)   # akin to sapply(df, sd) in R
##################################################

## two-way contingency table of categorical 
## outcome and predictors we want
## to make sure there are not 0 cells
## xtabs(~admit + rank, data = mydata)
table = pd.pivot_table(df, rows=['admit'], 
	cols=['rank'], aggfunc=len)

