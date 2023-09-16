from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix
import matplotlib.pyplot as plt
import pickle

with open('credit.pkl', 'rb') as f:
    X_credit_trainment, Y_credit_trainment, X_credit_test, Y_credit_test = pickle.load(f)


print(X_credit_trainment.shape, Y_credit_trainment.shape) # (1500, 3); (1500,) = (1500, 1)
print(X_credit_test.shape, Y_credit_test.shape) # (500, 3); (500,) = (500, 1)

tree_credit = DecisionTreeClassifier(criterion='entropy', random_state=0) # random_state == 0: It will always generate the same result
tree_credit.fit(X_credit_trainment, Y_credit_trainment)

forecast = tree_credit.predict(X_credit_test)
print(forecast) # [1 0 0 0...]

# Compare the forecast and the results
accuracy = accuracy_score(Y_credit_test, forecast)
print(accuracy)

cm = ConfusionMatrix(tree_credit)
cm.fit(X_credit_trainment, Y_credit_trainment)
cm.score(X_credit_test, Y_credit_test)
cm.show()

print(classification_report(Y_credit_test, forecast))

predictors = ['income', 'age', 'loan']
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(20, 20))
tree.plot_tree(tree_credit, feature_names=predictors, class_names=[str(c) for c in tree_credit.classes_], filled=True)
fig.savefig('credit_tree.png')