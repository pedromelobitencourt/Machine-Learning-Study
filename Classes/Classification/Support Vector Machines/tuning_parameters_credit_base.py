from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import pickle
import numpy as np

with open('credit.pkl', 'rb') as f:
    X_trainment, Y_trainment, X_test, Y_test = pickle.load(f)

X = np.concatenate((X_trainment, X_test))
Y = np.concatenate((Y_trainment, Y_test))

parameters = {'C': [1.0, 1.5, 2.0],
              'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
              'tol': [0.001, 0.0001, 0.00001]}

grid_search = GridSearchCV(estimator=SVC(), param_grid=parameters) # It will test the combinations, it's slow
grid_search.fit(X_trainment, Y_trainment)

best_parameters = grid_search.best_params_
best_result = grid_search.best_score_

print(best_parameters)
print(best_result)