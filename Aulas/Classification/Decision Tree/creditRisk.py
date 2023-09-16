import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import pickle

# It was already preprocessed (LabelEncoder...)
with open('credit_risk.pkl', 'rb') as f:
    X_credit_risk, Y_credit_risk = pickle.load(f)

tree_credit_risk = DecisionTreeClassifier(criterion='entropy')
tree_credit_risk.fit(X_credit_risk, Y_credit_risk)

print(tree_credit_risk.feature_importances_) # income > credit report > debt > guarantee
class_names = list(tree_credit_risk.classes_)

# Display the tree plot
plt.figure(figsize=(12, 6))
tree.plot_tree(tree_credit_risk, feature_names=["Credit Report", "Debt", "Guarantee", "Income"], filled=True, rounded=True, 
               class_names=class_names)

# If it satisfies the condition, it goes to the left. Otherwise, it goes to the right
plt.show()


forecast = tree_credit_risk.predict([[0, 0, 1, 2], [2, 0, 0, 0]])
print(forecast)