# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pylab as pl

import statsmodels.api as sm

from sklearn.metrics import roc_curve, auc
from sklearn.ensemble.gradient_boosting import LogOddsEstimator, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle

# <codecell>

url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data'
variables = ['Status of existing checking account',
             'Duration in month',
             'Credit history',
             'Purpose',
             'Credit amount',
             'Savings account/bonds',
             'Present employment since',
             'Installment rate in percentage of disposable income',
             'Personal status and sex',
             'Other debtors / guarantors',
             'Present residence since',
             'Property',
             'Age in years',
             'Other installment plans',
             'Housing',
             'Number of existing credits at this bank',
             'Job',
             'Number of people being liable to provide maintenance for',
             'Telephone',
             'foreign worker',
             'Classification']
df = pd.read_csv(url, sep=' ', names=variables)
df.head()

# <codecell>

to_snake = lambda x: x.lower().replace(" ", "_")

# <codecell>

df.columns = map(to_snake, df.columns)

# <codecell>

df.groupby(['classification']).size()

# <codecell>

df['bad'] = (df.classification==2).astype(int)
df = df.drop(['classification'], axis=1)

# <codecell>

df.groupby(['bad']).size()

# <codecell>

possible_features = np.array([col for col in df.columns if col != 'bad'])
possible_features

# <codecell>

grouper = ['purpose']

def prop_bad(group):
    nrow = float(len(group))
    n_bad = group.bad.sum()
    return n_bad / nrow

print df.groupby(grouper).apply(prop_bad)

# <codecell>

purpose_int = pd.factorize(df['purpose'], sort=True)[0]

df['purpose_int'] = df.purpose.replace(df.purpose, purpose_int)

# <codecell>

other_debtors_int = pd.factorize(df['other_debtors_/_guarantors'], sort=True)[0]

df['other_debtors_int'] = df['other_debtors_/_guarantors'].replace(df['other_debtors_/_guarantors'], other_debtors_int)

# <codecell>

np.all(df.groupby('other_debtors_/_guarantors').size() == df.groupby('other_debtors_int').size())

# <codecell>

credit_history_int = pd.factorize(df.credit_history, sort=True)[0]
df['credit_history_int'] = df.credit_history.replace(df.credit_history, credit_history_int)

# <codecell>

housing_int = pd.factorize(df.housing, sort=True)[0]
df['housing_int'] = df.housing.replace(df.housing, housing_int)

# <codecell>

df.age_in_years = df.age_in_years.astype(float)

# <codecell>

target = ['bad']
possible_features = ['credit_history_int', 'purpose_int', 'housing_int', 'age_in_years', 'other_debtors_int', 'number_of_existing_credits_at_this_bank']
#possible_features = ['age_in_years', 'credit_history_int']
selected = possible_features + target

# <codecell>

df2 = df[selected]

# <codecell>

X, y = shuffle(df2[possible_features], df2.bad)
offset = int(X.shape[0] * 0.9)
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]

# <codecell>

params = {'init': LogOddsEstimator(), 'n_estimators': 5, 'max_depth': 6, 'learning_rate': 0.1, 'loss': 'bdeviance'}
clf = GradientBoostingClassifier(**params)

# <codecell>

clf = clf.fit(X_train, y_train)
predicted = clf.predict(X_test)

# <codecell>

clf.feature_importances_

# <codecell>

print "Mean Squared Error"
mse = mean_squared_error(y_test, predicted)
print("MSE: %.4f" % mse)
print 

# <codecell>

params = clf.get_params()
params

# <codecell>

test_score = np.zeros((params['n_estimators'],), dtype=np.float64)
test_score

# <codecell>

for i, y_pred in enumerate(clf.staged_decision_function(X_test)):
    test_score[i] = clf.loss_(y_test, y_pred)
    
test_score

# <codecell>

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, clf.train_score_, 'b-',
        label='Training Set Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
        label='Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')

feature_importance = clf.feature_importances_

feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5

plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')

plt.yticks(pos, np.array(possible_features)[sorted_idx])

plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()

# <codecell>

df2 = sm.add_constant(df2, prepend=True) if not 'const' in df2.columns else df2

# <codecell>

test_train_split = 0.3

feature_names = ['credit_history_int', 'age_in_years']
df2['istest'] = np.random.uniform(size=len(df2)) < test_train_split

df2.groupby('istest').size()

# <codecell>

test = df2[df2.istest]
train = df2[-df2.istest]

# <codecell>

train.head()

# <codecell>

binomial = sm.families.Binomial()
model = sm.GLM.from_formula("bad ~ const + I(credit_history_int == 4) + age_in_years.astype(float)", df=train, family=binomial)

# <codecell>

result = model.fit()

# <codecell>

print result.summary()

# <codecell>

plt.scatter(train.age_in_years, train.bad)
plt.scatter(train.age_in_years, result.fittedvalues, color='r')

# <codecell>

plt.scatter(train.age_in_years, train.bad)
plt.scatter(train.age_in_years, np.exp(result.fittedvalues)/(1 + np.exp(result.fittedvalues)), color='r')

# <codecell>

params = result.params
params

# <codecell>

intercept = params.ix['Intercept']
history4_coef = params.ix['I(credit_history_int == 4)[T.True]']
age_coef = params.ix['age_in_years.astype(float)']
mean_age = df.age_in_years.mean()

# <codecell>

print intercept + (age_coef * mean_age)
print result.params
print result.summary()

# <codecell>

def odds_to_probability(odds):
    return odds / (1 + odds)

# <codecell>

odds_at_35 = np.exp(intercept + age_coef * df.age_in_years.mean())
odds_at_36 = np.exp(intercept + age_coef * df.age_in_years.mean()+1)
odds_to_probability(odds_at_36) - odds_to_probability(odds_at_35)

# <codecell>

test['predicted'] = result.predict(test)
test.head()

# <codecell>

plt.scatter(test.age_in_years, test.bad)
plt.scatter(test.age_in_years, test.predicted, color='r')

# <markdowncell>

# <h1>Receiver operating characteristic (ROC)</h1>

# <codecell>

# Compute ROC curve and area the curve
fpr, tpr, thresholds = roc_curve(test.bad, test.predicted)
roc_auc = auc(fpr, tpr)
print "Area under the ROC curve : %f" % roc_auc

# Plot ROC curve
pl.clf()
pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
pl.plot([0, 1], [0, 1], 'k--')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel('False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic')
pl.legend(loc="lower right")
pl.show()

# <codecell>

df = sm.add_constant(df, prepend=True) if not 'const' in df.columns else df
df['istest'] = np.nan
df['predicted'] = result.predict(df)

# <codecell>

df[test.columns].head()

# <codecell>

# Compute ROC curve and area the curve
fpr, tpr, thresholds = roc_curve(df.bad, df.predicted)
roc_auc = auc(fpr, tpr)
print "Area under the ROC curve : %f" % roc_auc

# Plot ROC curve
pl.clf()
pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
pl.plot([0, 1], [0, 1], 'k--')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel('False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic')
pl.legend(loc="lower right")
pl.show()

# <markdowncell>

# <h1>Rank Plot</h1>

# <codecell>

df[['bad','const','age_in_years','predicted']].head()

# <codecell>

df['predicted_demidecile'] = np.floor(20*df.predicted.rank()/len(df))
df[['bad','age_in_years','predicted', 'predicted_demidecile']].head()

# <codecell>

df.groupby('predicted_demidecile').predicted.mean()

# <codecell>

df.groupby('predicted_demidecile').bad.mean()

# <codecell>

plt.plot(df.groupby('predicted_demidecile')['predicted'].mean(), color = 'r')
plt.plot(df.groupby('predicted_demidecile')['bad'].mean())

# <codecell>


# <codecell>


