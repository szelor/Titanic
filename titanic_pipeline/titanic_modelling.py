
from azureml.core import Run
import pandas
import argparse
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

# Get the experiment run context
run = Run.get_context()

# Get parameters
parser = argparse.ArgumentParser()
parser.add_argument('--regularization', type=float, dest='reg_rate', default=0.05, help='regularization rate')
parser.add_argument('--data_folder', type=str, dest='data_folder', default="data", help='data location')
parser.add_argument('--output_folder', type=str, dest='output_folder', default="titanic_model", help='output folder')
args = parser.parse_args()
reg = args.reg_rate
data_folder = args.data_folder
output_folder = args.output_folder

# load prepared data
data_file = data_folder + "/prepared_data.csv"
print("Loading Data from " + data_file)
train_df = pandas.read_csv(data_file)

# Separate features and labels
X = train_df.drop("Survived", axis=1)
y = train_df["Survived"]

# Split data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12345)

# Train a logistic regression model
print('Training a logistic regression model with regularization rate of', reg)
run.log('Regularization Rate',  np.float(reg))
model = LogisticRegression(C=1/reg, solver="liblinear").fit(X_train, y_train)

# calculate accuracy
y_hat = model.predict(X_test)
acc = np.average(y_hat == y_test)
print('Accuracy:', acc)
run.log('Accuracy', np.float(acc))

# calculate AUC
y_scores = model.predict_proba(X_test)
auc = roc_auc_score(y_test,y_scores[:,1])
print('AUC: ' + str(auc))
run.log('AUC', np.float(auc))

# Save the trained model in the outputs folder
os.makedirs(output_folder, exist_ok=True)
output_path = output_folder + "/titanic_model.pkl"
joblib.dump(value=model, filename=output_path)

# Complete the run
run.complete()
