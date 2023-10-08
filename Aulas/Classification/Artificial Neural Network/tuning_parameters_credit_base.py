from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
import pickle
import numpy as np

with open('credit.pkl', 'rb') as f:
    X_trainment, Y_trainment, X_test, Y_test = pickle.load(f)

X = np.concatenate((X_trainment, X_test))
Y = np.concatenate((Y_trainment, Y_test))

parameters = {'activation': ['relu', 'logistic', 'tanh'],
              'solver': ['adam', 'sgd'],
              'batch_size': [10, 56]}

grid_search = GridSearchCV(estimator=MLPClassifier(), param_grid=parameters) # It will test the combinations, it's slow
grid_search.fit(X_trainment, Y_trainment)

best_parameters = grid_search.best_params_
best_result = grid_search.best_score_

print(best_parameters)
print(best_result)