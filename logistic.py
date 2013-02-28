import numpy as np
import pandas as pd
import pylab as pl
import statsmodels.api as sm
from scipy import stats
from pyroc import plot_multiple_roc, ROCData
from sklearn.cross_validation import train_test_split

df = pd.read_csv("data/copenhagen.csv")
df.head()



df.groupby(['housing', 'satisfaction']).size()
df.groupby(['influence', 'satisfaction']).size()
df.groupby(['contact', 'satisfaction']).size()

# define a feature set
features = ['influence']
# extract only the indepdendent variables
housing, _, _ = pd.factorize(df.housing)
influence, _, _ = pd.factorize(df.influence)
contact, _, _ = pd.factorize(df.contact)

X = pd.DataFrame({
    "housing": housing,
    "influence": influence,
    "contact": contact
    })

# adding a constant for the intercept
X = sm.add_constant(X, prepend=False)
y = df['satisfaction'].apply(lambda x: 0 if x=="low" else 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.8)

print y_train
# create a GLM
glm = sm.GLM(y_train, X_train, family=sm.families.Binomial())
# fit the model
res = glm.fit()
# take a look at the output
print res.summary()
print res.params
print res.conf_int()
print res.aic

# generate an ROC curve
roc = ROCData(zip(y_test, res.predict(X_test)))
roc.auc()
print roc
roc.plot(title='ROC Curve', include_baseline=True)

# plot_multiple_roc(rocs,'Multiple ROC Curves',include_baseline=True)
