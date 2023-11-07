from sklearn.svm import SVC
import numpy as np
import pickle

with open('credit.pkl', 'rb') as f:
    X_trainment, Y_trainment, X_test, Y_test = pickle.load(f)

X = np.concatenate((X_trainment, X_test))
Y = np.concatenate((Y_trainment, Y_test))

# After all analysis

svm_classifier = SVC(C=2.0, kernel='rbf', probability=True)
# Final model: We can train all data base
svm_classifier.fit(X, Y)

pickle.dump(svm_classifier, open('final_trained_svm_classifier.sav', 'wb')) # Saving the classifier