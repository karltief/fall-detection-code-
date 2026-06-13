import time
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, jaccard_score
from sklearn.model_selection import train_test_split, GridSearchCV
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

file_path = r"C:\Users\UOE\PycharmProjects\ML_Fall\output_m2.csv"
df = pd.read_csv(file_path)

# Label the columns for the classification processes
X = df[df.columns[0:103]]
y = df["Fall/NoFALL(1,0)"].values.ravel()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

# Define your coarse grid with the specified C values
C_values = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 500, 1000, 1500]
param_grid = {'C': C_values}

# Initialize Logistic Regression Model
log_reg = LogisticRegression(solver='liblinear')

# Use GridSearchCV to perform a coarse grid search (with cv=3)
grid_search = GridSearchCV(log_reg, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1, return_train_score=True)

# Measure training time for coarse grid search
start_train = time.time()
grid_search.fit(X_train, y_train)
end_train = time.time()
train_time = end_train - start_train

best_model = grid_search.best_estimator_
best_C = grid_search.best_params_['C']

# Extract train and test accuracy scores from the coarse grid search
coarse_train_scores = grid_search.cv_results_['mean_train_score']
coarse_test_scores = grid_search.cv_results_['mean_test_score']

start_pred = time.time()
y_pred = best_model.predict(X_test)
end_pred = time.time()
pred_time = end_pred - start_pred

# Compute evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
jaccard = jaccard_score(y_test, y_pred, pos_label=1)

print(f"\nBest C Value Found: {best_C}")
print(f"Training Time: {train_time:.4f} seconds")
print(f"Prediction Time: {pred_time:.4f} seconds")
print(f"Model Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Jaccard Score: {jaccard:.4f}")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(10, 6))
plt.plot(C_values, coarse_train_scores, marker='o', linestyle='-', color='blue', label="Training Accuracy")
plt.plot(C_values, coarse_test_scores, marker='s', linestyle='-', color='green', label="Test Accuracy")
plt.xscale('log')  # Log scale for better visualization across the wide range of C values
plt.xlabel("C Value (Log Scale)")
plt.ylabel("Accuracy")
plt.title("Logistic Regression Accuracy vs. C Value for Dataset 2 (Grid Search)")
plt.legend()
plt.grid(True)

# Identify the coarse best generalization point (where train and test accuracy are closest)
coarse_diff = np.abs(np.array(coarse_train_scores) - np.array(coarse_test_scores))
min_diff_index = np.argmin(coarse_diff)
best_C_overlap = C_values[min_diff_index]
overlap_accuracy = (coarse_train_scores[min_diff_index] + coarse_test_scores[min_diff_index]) / 2

plt.annotate(
    f"Best Gen\nC={best_C_overlap:.4f}, Acc={overlap_accuracy:.4f}",
    xy=(best_C_overlap, overlap_accuracy),
    xycoords='data',
    xytext=(20, -50),
    textcoords='offset points',
    arrowprops=dict(facecolor='red', arrowstyle="->"),
    fontsize=12,
    color='red',
    zorder=10,
    clip_on=False
)

# Define a refined grid with 10 values linearly spaced around the coarse best generalization point
refined_C_values = np.linspace(best_C_overlap*0.5, best_C_overlap * 5, 10)
param_grid_refined = {'C': refined_C_values}

# Use GridSearchCV to perform the refined grid search (with cv=5 for more stability)
refined_grid_search = GridSearchCV(log_reg, param_grid_refined, cv=5, scoring='accuracy', n_jobs=-1, verbose=1, return_train_score=True)
refined_grid_search.fit(X_train, y_train)

# Extract refined accuracy scores for training and test sets
refined_train_scores = refined_grid_search.cv_results_['mean_train_score']
refined_test_scores = refined_grid_search.cv_results_['mean_test_score']

# Identify the refined best generalization point
refined_diff = np.abs(np.array(refined_train_scores) - np.array(refined_test_scores))
min_diff_index_refined = np.argmin(refined_diff)
refined_best_C_overlap = refined_C_values[min_diff_index_refined]
refined_overlap_accuracy = (refined_train_scores[min_diff_index_refined] + refined_test_scores[min_diff_index_refined]) / 2

print(f"\nRefined Best Generalization Point: C = {refined_best_C_overlap:.4f} with Avg. Accuracy = {refined_overlap_accuracy:.4f}")

inset_ax = inset_axes(plt.gca(), width="40%", height="30%", loc="lower right", borderpad=3)
inset_ax.plot(refined_C_values, refined_train_scores, marker='o', linestyle='-', color='blue', label="Train")
inset_ax.plot(refined_C_values, refined_test_scores, marker='s', linestyle='-', color='green', label="Test")
# Set linear x-axis limits for the narrow refined range
inset_ax.set_xlim(refined_C_values[0], refined_C_values[-1])
inset_ax.set_ylim(refined_overlap_accuracy - 0.03, refined_overlap_accuracy + 0.03)
inset_ax.grid(True)
inset_ax.set_title("Zoomed In", fontsize=10)

# Annotate the refined best generalization point within the inset
inset_ax.annotate(
    f"Refined Best Gen\nC={refined_best_C_overlap:.4f}\nAcc={refined_overlap_accuracy:.4f}",
    xy=(refined_best_C_overlap, refined_overlap_accuracy),
    xycoords='data',
    xytext=(10, 70),
    textcoords='offset points',
    arrowprops=dict(facecolor='red', arrowstyle="->"),
    fontsize=10,
    color='red',
    clip_on=False
)


plt.savefig("logreg_accuracy_vs_C_with_refined_zoom_m2.png", bbox_inches='tight')
plt.show()
