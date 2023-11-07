from sklearn.neural_network import MLPClassifier
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

neural_network_results = []

for i in range(30):
    kfold = KFold(n_splits=10, shuffle=True, random_state=i)

    neural_network = MLPClassifier(activation='relu', batch_size=56, solver='adam') # parameters got from tuning
    
    scores = cross_val_score(neural_network, X, Y, cv=kfold)
    # print(scores)
    # print(scores.mean(), '\n')

    neural_network_results.append(scores.mean())

results = pd.DataFrame({'SVM': neural_network_results})

print(neural_network_results, '\n')
print(results.describe(), '\n')
print(results.var(), '\n')
print((results.std() / results.mean()) * 100, '\n')

alpha = 0.05

print(shapiro(neural_network_results)) # if pvalue < alpha: It rejects the null hypothesis (it's not in a normal distribution). Otherwise, it accepts it
sns.displot(neural_network_results, kind='kde')
plt.show()