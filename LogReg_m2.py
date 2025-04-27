from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
from functions import *
from sklearn import metrics
import joblib
import seaborn as sns

file_path = r"C:\Users\UOE\PycharmProjects\ML_Fall\output_m2.csv"
file_path1 = r"C:\Users\UOE\PycharmProjects\ML_Fall\output.csv"
df = pd.read_csv(file_path)

# Label the columns for the classification process
X = df[df.columns[0:103]]
y = df[["Fall/NoFALL(1,0)"]].values.ravel()
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)
C = [0.01, 0.1,0.5,1,5,10,50,100,500,1000]
C = [1]
for C in C:
    start_train = time.time()
    LR = LogisticRegression(C=0.003, solver ='liblinear').fit(X_train,y_train)
    end_train = time.time()
    train_time = end_train - start_train

    start_predict = time.time()
    y_pred = LR.predict(X_test)
    end_predict = time.time()
    predict_time = end_predict - start_predict

    # GridSearch and hyper-parameter tuning
    param = {'C': np.logspace(-3, 2, 25)}
    grid = GridSearchCV(LR, param_grid=param, cv=3)
    grid.fit(X_train, y_train)
    print(grid.best_params_, grid.cv_results_)
    plot_c_score(grid, "logreg_meth2", "Logistic Regression", "2", -1, 20)

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred, labels=[0,1])
    np.set_printoptions(precision=2)
    print(accuracy_score(y_test, y_pred))
    print (classification_report(y_test, y_pred))

    # Plot non-normalized confusion matrix
    plot_confusion_matrix(cnf_matrix, classes=['Fall(1)', 'Not Fall(0)'], normalize=True,
                          title='Confusion matrix - Logistic Regression (Dataset 2)')

    # Extract feature names and corresponding weights
    feature_names = X.columns
    feature_weights = LR.coef_[0]

    # Create a DataFrame for better visualization
    weights_df = pd.DataFrame({"Feature": feature_names, "Weight": feature_weights})
    weights_df = weights_df.sort_values(by="Weight", ascending=False)

    print(f'Training Time: {train_time:.4f}s')
    print(f'Predict Time: {predict_time:.4f}s')

    plt.figure(figsize=(20, 6))
    sns.barplot(x=weights_df["Feature"], y=(weights_df["Weight"]/5), palette="coolwarm")
    plt.xticks(rotation=90)
    plt.xlabel("Feature Name")
    plt.ylabel("Weight (Importance)")
    plt.title("Feature Importance in Logistic Regression (Fall Detection) when C = 1")
    plt.savefig(f"weighted_logreg/weightsC={C}.png", bbox_inches='tight')
    plt.show(bbox_inches='tight')

# Save SVM model
    joblib.dump(LR, 'LogReg_model.pkl')