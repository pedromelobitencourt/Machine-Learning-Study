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

print(neural_network_answer, svm_answer, tree_answer)

answers = [neural_network_answer[0], svm_answer[0], tree_answer[0]]
print(answers)

default = 0
pay = 0

for answer in answers:
    if answer:
        default += 1
    else:
        pay += 1

if default > pay:
    print('the client will default the loan')
elif default == pay:
    print('Tie! I cant conclude it')
else:
    print('The client will pay the loan')