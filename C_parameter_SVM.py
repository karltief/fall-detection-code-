from sklearn import svm
import pandas as pd
import numpy as np
import joblib
import time
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, f1_score, jaccard_score, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

df = pd.read_csv("../output_m2.csv")
X = df[df.columns[0:103]]  # Features
y = df["Fall/NoFALL(1,0)"].values.ravel()  # Target labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

# Define coarse grid for C values: 10 values logarithmically spaced between 10^-3 and 10^3
coarse_C_values = np.logspace(-3, 3, 10)
param_grid_coarse = {'C': coarse_C_values}

# Initialize and fit SVM Model using GridSearchCV with cv=3
svm_model = svm.SVC(kernel='rbf')
grid_search = GridSearchCV(svm_model, param_grid_coarse, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)

start_train = time.time()
grid_search.fit(X_train, y_train)
end_train = time.time()
train_time = end_train - start_train

# Get the best SVM model and best C value from coarse grid search (cv=3)
best_svm = grid_search.best_estimator_
best_C = grid_search.best_params_['C']
print(f"\nBest C Value Found: {best_C}")
print(f"Training Time for coarse grid search: {train_time:.4f} seconds")

# Measure prediction time on test data
start_pred = time.time()
yhat = best_svm.predict(X_test)
end_pred = time.time()
pred_time = end_pred - start_pred

# Compute evaluation metrics on test set
f1 = f1_score(y_test, yhat, average='weighted')
jaccard = jaccard_score(y_test, yhat, pos_label=1)
accuracy = accuracy_score(y_test, yhat)

print("\nClassification Report:")
print(classification_report(y_test, yhat))
print(f"\nPrediction Time: {pred_time:.4f} seconds")
print(f"Model Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Jaccard Score: {jaccard:.4f}")

# Save the best SVM model from the coarse grid search
joblib.dump(best_svm, f'../best_svm_model_C={best_C}.pkl')

grid_search = GridSearchCV(svm_model, param_grid_coarse, cv=3, scoring='accuracy', n_jobs=-1, verbose=1, return_train_score=True)
grid_search.fit(X_train, y_train)

# Extract training and test accuracy scores
coarse_train_scores = grid_search.cv_results_['mean_train_score']
coarse_test_scores = grid_search.cv_results_['mean_test_score']

# Find the coarse generalization point where the difference between train and test is smallest
coarse_diff = np.abs(np.array(coarse_train_scores) - np.array(coarse_test_scores))
min_diff_index = np.argmin(coarse_diff)
best_C_overlap = coarse_C_values[min_diff_index]
overlap_accuracy = (coarse_train_scores[min_diff_index] + coarse_test_scores[min_diff_index]) / 2

print(f"\nCoarse Best Generalization Point: C = {best_C_overlap:.4f} with Avg. Accuracy = {overlap_accuracy:.4f}")

# Define a refined grid: 10 values linearly spaced between 0.8*best_C_overlap and 1.2*best_C_overlap
refined_C_values = np.linspace(best_C_overlap * 0.5, best_C_overlap * 5, 10)
param_grid_refined = {'C': refined_C_values}

refined_grid_search = GridSearchCV(svm_model, param_grid_refined, cv=3, scoring='accuracy', n_jobs=-1, verbose=1, return_train_score=True)
refined_grid_search.fit(X_train, y_train)

# Extract refined training and test accuracy scores
refined_train_scores = refined_grid_search.cv_results_['mean_train_score']
refined_test_scores = refined_grid_search.cv_results_['mean_test_score']

# Find the refined generalization point
refined_diff = np.abs(np.array(refined_train_scores) - np.array(refined_test_scores))
min_diff_index_refined = np.argmin(refined_diff)
refined_best_C_overlap = refined_C_values[min_diff_index_refined]
refined_overlap_accuracy = (refined_train_scores[min_diff_index_refined] + refined_test_scores[min_diff_index_refined]) / 2

print(f"\nRefined Best Generalization Point: C = {refined_best_C_overlap:.4f} with Avg. Accuracy = {refined_overlap_accuracy:.4f}")

fig, ax = plt.subplots(figsize=(10, 6))
# Plot coarse grid results
ax.plot(coarse_C_values, coarse_train_scores, marker='o', linestyle='-', color='blue', label="Training Accuracy")
ax.plot(coarse_C_values, coarse_test_scores, marker='s', linestyle='-', color='green', label="Test Accuracy")
ax.set_xscale('log')
ax.set_xlabel("C Value (Log Scale)")
ax.set_ylabel("Accuracy")
ax.set_title("SVM Accuracy vs. C Value for Dataset 2 (Grid Search)")
ax.legend()
ax.grid(True)

# Annotate this coarse best generalization point
plt.annotate(
    f"Best Gen\nC={best_C_overlap:.4f}, Acc={overlap_accuracy:.4f}",
    xy=(best_C_overlap, overlap_accuracy),
    xycoords='data',
    xytext=(10, 150),
    textcoords='offset points',
    arrowprops=dict(facecolor='red', arrowstyle="->"),
    fontsize=12,
    color='red',
    zorder=10,
    clip_on=False
)

# Create an inset axis to display the refined grid search results
inset_ax = inset_axes(ax, width="40%", height="30%", loc="lower right", borderpad=3)
inset_ax.plot(refined_C_values, refined_train_scores, marker='o', linestyle='-', color='blue', label="Train")
inset_ax.plot(refined_C_values, refined_test_scores, marker='s', linestyle='-', color='green', label="Test")
# For the refined grid, using a linear scale since the interval is narrow
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

plt.savefig("svm_accuracy_vs_C_train_test_with_refined_zoom_m2.png", bbox_inches='tight')
plt.show()
