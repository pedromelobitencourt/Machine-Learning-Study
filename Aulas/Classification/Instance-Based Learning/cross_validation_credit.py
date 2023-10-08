from sklearn.neighbors import KNeighborsClassifier
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

knn_results = []

for i in range(30):
    kfold = KFold(n_splits=10, shuffle=True, random_state=i)

    knn = KNeighborsClassifier() # parameters got from tuning
    
    scores = cross_val_score(knn, X, Y, cv=kfold)
    # print(scores)
    # print(scores.mean(), '\n')

    knn_results.append(scores.mean())

results = pd.DataFrame({'SVM': knn_results})

print(knn_results, '\n')
print(results.describe(), '\n')
print(results.var(), '\n')

print((results.std() / results.mean()) * 100, '\n')

alpha = 0.05

print(shapiro(knn_results)) # if pvalue < alpha: It rejects the null hypothesis (it's not in a normal distribution). Otherwise, it accepts it
sns.displot(knn_results, kind='kde')
plt.show()