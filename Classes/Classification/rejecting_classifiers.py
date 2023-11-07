# After implementing and analysing some machine learning algorithms and choosing the best ones, we will combine their answers
# Best ones: Neural network, SVM and Decision Tree

import numpy as np
import pickle

with open('credit.pkl', 'rb') as f:
    X_trainment, Y_trainment, X_test, Y_test = pickle.load(f)

X = np.concatenate((X_trainment, X_test))
Y = np.concatenate((Y_trainment, Y_test))

new_data = X[0]
new_data = new_data.reshape(1, -1)

neural_network = pickle.load(open('final_trained_neural_network_classifier.sav', 'rb'))
svm = pickle.load(open('final_trained_svm_classifier.sav', 'rb'))
tree = pickle.load(open('final_trained_tree_classifier.sav', 'rb'))

neural_network_answer = neural_network.predict(new_data)
svm_answer = neural_network.predict(new_data)
tree_answer = neural_network.predict(new_data)

neural_network_probability = neural_network.predict_proba(new_data)
neural_network_trust = neural_network_probability.max()

tree_probability = tree.predict_proba(new_data)
tree_trust = tree_probability.max()

svm_probability = svm.predict_proba(new_data)
svm_trust = svm_probability.max()

print(neural_network_probability, neural_network_trust)
print(svm_probability, svm_trust)
print(tree_probability, tree_trust)

answers = [neural_network_answer[0], svm_answer[0], tree_answer[0]]
trusts = [neural_network_trust, svm_trust, tree_trust]

default = 0
pay = 0

min_trust = 0.999999
used_algorithms = 0

for i in range(len(trusts)):
    if trusts[i] > min_trust:
        if answers[i]:
            default += 1
        else:
            pay += 1
        used_algorithms += 1

if default > pay:
    print('the client will default the loan')
elif default == pay:
    print('Tie! I cant conclude it')
else:
    print('The client will pay the loan')

print(pay, default)
print(used_algorithms)