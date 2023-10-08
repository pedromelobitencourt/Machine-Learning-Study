from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import pickle
import numpy as np

with open('credit.pkl', 'rb') as f:
    X_trainment, Y_trainment, X_test, Y_test = pickle.load(f)

X = np.concatenate((X_trainment, X_test))
Y = np.concatenate((Y_trainment, Y_test))

parameters = {'tol': [0.0001, 0.00001, 0.000001],
              'C': [1.0, 1.5, 2.0],
              'solver': ['lbfgs', 'sag', 'saga']}

grid_search = GridSearchCV(estimator=LogisticRegression(), param_grid=parameters) # It will test the combinations, it's slow
grid_search.fit(X_trainment, Y_trainment)

best_parameters = grid_search.best_params_
best_result = grid_search.best_score_

print(best_parameters)
print(best_result)