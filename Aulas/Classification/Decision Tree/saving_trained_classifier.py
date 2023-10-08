from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pickle

with open('credit.pkl', 'rb') as f:
    X_trainment, Y_trainment, X_test, Y_test = pickle.load(f)

X = np.concatenate((X_trainment, X_test))
Y = np.concatenate((Y_trainment, Y_test))

# After all analysis

tree_classifier = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=1, min_samples_split=5, splitter='best')
tree_classifier.fit(X, Y)

# Final model: We can train all data base
pickle.dump(tree_classifier, open('final_trained_tree_classifier.sav', 'wb')) # Saving the classifier