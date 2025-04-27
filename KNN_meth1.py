from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import metrics
import joblib
from sklearn.metrics import confusion_matrix, classification_report

from functions import *


#df = pd.read_csv("ML_Fall/Method_1/output_m1WBoundary.csv")
file_path = r"C:\Users\UOE\PycharmProjects\ML_Fall\Method_1\output_m1WoBoundary.csv"
file_path1 = r"C:\Users\UOE\PycharmProjects\ML_Fall\output.csv"
df = pd.read_csv(file_path)
# Label the columns for the classification process
X = df[df.columns[0:103]]
print(X.columns)

y = df[["Fall/NoFALL(1,0)"]].values.ravel()
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

#Model
Ks = 12
mean_acc = np.zeros((Ks - 1))
std_acc = np.zeros((Ks - 1))

for n in range(1, Ks):
    # Train Model and Predict
    neigh = KNeighborsClassifier(n_neighbors=n).fit(X_train, y_train)
    yhat = neigh.predict(X_test)
    mean_acc[n - 1] = metrics.accuracy_score(y_test, yhat)
    std_acc[n - 1] = np.std(yhat == y_test) / np.sqrt(yhat.shape[0])

# Testing accuracy
start_train = time.time()
neighmax = KNeighborsClassifier(n_neighbors=6).fit(X_train, y_train)
end_train = time.time()
train_time = end_train - start_train
start_predict = time.time()
yhat = neighmax.predict(X_test)
end_predict = time.time()
predict_time = end_predict-start_predict
print(f'Training Time: {train_time:.4f}s')
print(f'Predict Time: {predict_time:.4f}s')
print(yhat[0:5])
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))
print(metrics.classification_report(y_test, yhat, digits = 4))


plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.fill_between(range(1,Ks),mean_acc - 3 * std_acc,mean_acc + 3 * std_acc, alpha=0.10,color="green")
plt.legend(('Accuracy ', '+/- 1xstd','+/- 3xstd'), fontsize='medium')
plt.ylabel('Accuracy')
plt.title("Accuracy of the model for each KNN value for Dataset 1")
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.savefig("KNN_maxAccuracy_meth1.png")
plt.show()

print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1)
cnf_matrix = confusion_matrix(y_test, yhat, labels=[0,1])
plot_confusion_matrix(cnf_matrix, classes=['Fall(1)', 'Not Fall(0)'], normalize=True, title='Confusion matrix - KNN (Dataset 1)')


cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
print(np.round(cnf_matrix,2))
# saving max accuracy K model
best_knn = KNeighborsClassifier(n_neighbors=(mean_acc.argmax()+1)).fit(X_train, y_train)

joblib.dump(best_knn, "../best_knn_model.pkl")
print("Best model saved as 'best_knn_model.pkl'")