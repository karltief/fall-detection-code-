from sklearn import svm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, jaccard_score, accuracy_score
import itertools
from functions import *

file_path = r"C:\Users\UOE\PycharmProjects\ML_Fall\Method_1\output_m1WoBoundary.csv"
file_path1 = r"C:\Users\UOE\PycharmProjects\ML_Fall\Method_1\output_m1.csv"
df = pd.read_csv(file_path)

# Label the columns for the classification process
X = df[df.columns[0:103]]
y = df[["Fall/NoFALL(1,0)"]].values.ravel()
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)
#Model
start_train = time.time()
clf = svm.SVC(kernel='rbf', C = 0.1)

clf.fit(X_train, y_train)
end_train = time.time()
train_time = end_train - start_train
start_predict = time.time()
yhat = clf.predict(X_test)
end_predict = time.time()
predict_time = end_predict-start_predict
print(f'Training Time: {train_time:.4f}s')
print(f'Predict Time: {predict_time:.4f}s')

# GridSearch and hyper parameter tuning
param = {'C':np.logspace(-3,2, 25 )}
grid = GridSearchCV(clf, param_grid=param, cv=3)
grid.fit(X_train, y_train)
print(grid.best_params_, grid.cv_results_)
plot_c_score(grid, "svm_meth1", "SVM", "1", -1, 30)


# Compute confusion matrix and Classification Report
cnf_matrix = confusion_matrix(y_test, yhat, labels=[0,1])
np.set_printoptions(precision=2)
print (classification_report(y_test, yhat))
print(accuracy_score(y_test, yhat))
# Compute the Mutual Information Bar Chart and Array
mutual_info_plot(mutual_info(X,y))
# Plot non-normalized confusion matrix
plot_confusion_matrix(cnf_matrix, classes=['Fall(1)', 'Not Fall(0)'], normalize=True, title='Confusion matrix - SVM (Dataset 1)')

# f1 Score
f1_score(y_test, yhat, average='weighted')
# Jaccard Score
jaccard_score(y_test, yhat, pos_label=1)

# Save SVM model
joblib.dump(clf, '../svm_model.pkl')