import pandas as pd # Load csv files
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split # Split the dataset into training and test sets
from sklearn.naive_bayes import GaussianNB # Naive Bayes
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from yellowbrick.classifier import ConfusionMatrix
import pickle # Save the trainment and test sets

with open('census.pkl', 'rb') as f:
    X_census_trainment, Y_census_trainment, X_census_test, Y_census_test = pickle.load(f)

naive_census_data = GaussianNB()
naive_census_data.fit(X_census_trainment, Y_census_trainment)
forecast = naive_census_data.predict(X_census_test)
print(accuracy_score(Y_census_test, forecast))

cm = ConfusionMatrix(naive_census_data)
cm.fit(X_census_trainment, Y_census_trainment)
cm.score(X_census_test, Y_census_test)
cm.show()
print(classification_report(Y_census_test, forecast))