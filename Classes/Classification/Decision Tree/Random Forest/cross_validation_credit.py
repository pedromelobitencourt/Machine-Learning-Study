from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold
from scipy.stats import shapiro
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pickle

with open('credit.pkl', 'rb') as f:
    X_trainment, Y_trainment, X_test, Y_test = pickle.load(f)

X = np.concatenate((X_trainment, X_test))
Y = np.concatenate((Y_trainment, Y_test))

random_forest_results = []

for i in range(30):
    kfold = KFold(n_splits=10, shuffle=True, random_state=i)

    random_forest = RandomForestClassifier(criterion='entropy', min_samples_leaf=1, min_samples_split=5, n_estimators=10) # parameters got from tuning
    
    scores = cross_val_score(random_forest, X, Y, cv=kfold)
    # print(scores)
    # print(scores.mean(), '\n')

    random_forest_results.append(scores.mean())

results = pd.DataFrame({'SVM': random_forest_results})

print(random_forest_results, '\n')
print(results.describe(), '\n')
print(results.var(), '\n')
print((results.std() / results.mean()) * 100, '\n')

alpha = 0.05

print(shapiro(random_forest_results)) # if pvalue < alpha: It rejects the null hypothesis (it's not in a normal distribution). Otherwise, it accepts it
sns.displot(random_forest_results, kind='kde')
plt.show()