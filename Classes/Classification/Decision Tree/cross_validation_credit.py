from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, KFold
from scipy.stats import shapiro
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pickle

with open('credit.pkl', 'rb') as f:
    X_trainment, Y_trainment, X_test, Y_test = pickle.load(f)

X = np.concatenate((X_trainment, X_test))
Y = np.concatenate((Y_trainment, Y_test))

tree_results = []

for i in range(30):
    kfold = KFold(n_splits=10, shuffle=True, random_state=i)

    credit_tree = DecisionTreeClassifier(min_samples_leaf=1, min_samples_split=5, criterion='entropy', splitter='best') # parameters got from tuning
    
    scores = cross_val_score(credit_tree, X, Y, cv=kfold)
    # print(scores)
    # print(scores.mean(), '\n')

    tree_results.append(scores.mean())

results = pd.DataFrame({'SVM': tree_results})

print(tree_results, '\n')
print(results.describe(), '\n')
print(results.var(), '\n')
print((results.std() / results.mean()) * 100, '\n')

alpha = 0.05

print(shapiro(tree_results)) # if pvalue < alpha: It rejects the null hypothesis (it's not in a normal distribution). Otherwise, it accepts it
sns.displot(tree_results, kind='kde')
plt.show()