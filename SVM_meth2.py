from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, f1_score, jaccard_score, roc_curve, accuracy_score
from functions import *
from sklearn.inspection import permutation_importance
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# Source the array of pose landmarks and features
file_path = r"C:\Users\UOE\PycharmProjects\ML_Fall\output_m2.csv"
df = pd.read_csv(file_path)

# Label the columns for the classification process
X = df[df.columns[0:103]]
y = df[["Fall/NoFALL(1,0)"]].values.ravel()
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

# Grid Search for hyperparameter tuning and fitting the model; finding also the training time and prediction time
start_train = time.time()
clf = svm.SVC(kernel='rbf', C=9)
clf.fit(X_train, y_train)
end_train = time.time()
train_time = end_train - start_train
start_predict = time.time()
yhat = clf.predict(X_test)
end_predict = time.time()
predict_time = end_predict-start_predict
print(f'Training Time: {train_time:.4f}s')
print(f'Predict Time: {predict_time:.4f}s')
''' # GridSearch and hyper parameter tuning
param = {'C':np.logspace(-3,2, 25 )}
grid = GridSearchCV(clf, param_grid=param, cv=3)
grid.fit(X_train, y_train)
print(grid.best_params_, grid.cv_results_)
plot_c_score(grid, "svm_meth2", "SVM", "2", 0, 20)
'''

# Plot the confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat, labels=[0,1])
np.set_printoptions(precision=2)
print(accuracy_score(y_test, yhat))
print (classification_report(y_test, yhat))

# Plot non-normalized confusion matrix
plot_confusion_matrix(cnf_matrix, classes=['Fall(1)', 'Not Fall(0)'], normalize=True, title='Confusion matrix - SVM (Dataset 2)')

# f1 Score
f1_score(y_test, yhat, average='weighted')
# Plot mutual information bar graph
mutual_info_plot(mutual_info(X,y))
# Jaccard Score
jaccard_score(y_test, yhat, pos_label=1)
# Save SVM model
joblib.dump(clf, '../svm_m2_model.pkl')