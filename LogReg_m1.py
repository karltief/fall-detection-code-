from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np

from functions import *
from sklearn import metrics
import joblib
import seaborn as sns

file_path = r"C:\Users\UOE\PycharmProjects\ML_Fall\Method_1\output_m1WoBoundary.csv"
file_path1 = r"C:\Users\UOE\PycharmProjects\ML_Fall\Method_1\output_m1.csv"
df = pd.read_csv(file_path)
print(df.head())

# Label the columns for the classification process
X = df[df.columns[0:103]]
y = df[["Fall/NoFALL(1,0)"]].values.ravel()
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

start_train = time.time()
LR = LogisticRegression( solver ='liblinear', C=5).fit(X_train,y_train)
end_train = time.time()
train_time = end_train - start_train
print(LR)
start_predict = time.time()
y_pred = LR.predict(X_test)

end_predict = time.time()
predict_time = end_predict - start_predict
print(f'Training Time: {train_time:.4f}s')
print(f'Predict Time: {predict_time:.4f}s')
print(y_pred)

 # GridSearch and hyper parameter tuning
param = {'C':np.logspace(-4,4, 25 )}
grid = GridSearchCV(LR, param_grid=param, cv=3)
grid.fit(X_train, y_train)
print(grid.best_params_, grid.cv_results_)
plot_c_score(grid, "logreg_meth1", "Logistic Regression", "1", -1, 20)


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred, labels=[0,1])
np.set_printoptions(precision=2)

print("Test set Accuracy: ", metrics.accuracy_score(y_test, y_pred))
print (classification_report(y_test, y_pred))

# Plot non-normalized confusion matrix
plot_confusion_matrix(cnf_matrix, classes=['Fall(1)', 'Not Fall(0)'], normalize=True, title='Confusion matrix - Logistic Regression (Dataset 1)')


# Extract feature names and corresponding weights
feature_names = X.columns
feature_weights = LR.coef_[0]  # Logistic regression stores weights in coef_

# Create a DataFrame for better visualization
weights_df = pd.DataFrame({"Feature": feature_names, "Weight": feature_weights})
weights_df = weights_df.sort_values(by="Weight", ascending=False)  # Sort by importance
# Feature weights
plt.figure(figsize=(20, 6))
sns.barplot(x=weights_df["Feature"], y=(weights_df["Weight"] / 5), palette="coolwarm")
plt.xticks(rotation=90)
plt.xlabel("Feature Name")
plt.ylabel("Weight (Importance)")
plt.title("Feature Importance in Logistic Regression (Fall Detection) when C = 1")
plt.savefig(f"weightsC.png", bbox_inches='tight')
plt.show(bbox_inches='tight')
# Save SVM model
joblib.dump(LR, 'LogReg_model.pkl')