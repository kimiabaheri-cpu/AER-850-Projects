# Step 1: Data Processing
import pandas as pd
df = pd.read_csv("Project 1 Data.csv")
print(df.head())
print("Size of DataFrame:", df.shape)
print(df.info())
#-------------------------------------------------------------

# Step 2: Data Visualization
import numpy as np
import matplotlib.pyplot as plt

## Satistical Analysis 
print("Satistical Analysis:")
print(df.describe())
    
## Mean Values 
print("\nMean values per Step:")
print(df.groupby('Step').mean(numeric_only=True))

## Standard Deviation 
print("\nStandard deviation per Step:")
print(df.groupby('Step').std(numeric_only=True))

## Histogram of X,Y,Z group by Step
step_counts = df["Step"].value_counts().sort_index()
plt.figure(figsize=(8,6))
plt.bar(step_counts.index, step_counts.values, color="forestgreen", edgecolor="black")
plt.title("Number of Values in Each Step")
plt.xlabel("Step")
plt.ylabel("Count")
plt.xticks(step_counts.index)  # show all steps 1–13 on x-axis
plt.show()

## Boxplot of X, Y, Z by Step
df.boxplot(column=["X", "Y", "Z"], by='Step',figsize=(12,6))
plt.suptitle("Boxplot of Features by Step")
plt.show()

## Scatter Plot of the X, Y, and Z values in each step (2D)
fig, ax = plt.subplots(figsize=(8,6))
ax.scatter(df["Step"], df["X"], marker='o', label="X")
ax.scatter(df["Step"], df["Y"], marker='o', label="Y")
ax.scatter(df["Step"], df["Z"], marker='o', label="Z")
ax.set_title("Scatter Plot of the X, Y, and Z values in each step")
ax.set_xlabel("Step")
ax.set_ylabel("Value")
ax.legend(title="Variable")
plt.tight_layout()
plt.show()

## Scatter plot for X, Y, Z 3D Shape
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
for step in df["Step"].unique():
    subset = df[df["Step"] == step]
    ax.scatter(subset["X"], subset["Y"], subset["Z"], label=f"Step {step}", alpha=0.7)
ax.set_title("3D Shape Scatter Representation of the Steps")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()
plt.show()
#------------------------------------------------------------------------------------------------

# Step 3: Correlation Analysis
import seaborn as sns
corr_matrix = df.corr(method="pearson", numeric_only=True)
print("\nPearson Correlation Matrix:")
print(corr_matrix)

## Plot
plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", cbar=True, square=True)
plt.title("Correlation Heatmap (Pearson)")
plt.show()

## Target Correlation
print("\nCorrelation with Target (Step):")
print(corr_matrix["Step"].sort_values(ascending=False))
#---------------------------------------------------------------------------------------------------------

# Step 4: Classification Model Development
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
print("\nStep4: Classification Model Development")
x = df.drop(columns=["Step"])
y = df["Step"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)
scalar = StandardScaler()
x_train = scalar.fit_transform(x_train)
x_test = scalar.transform(x_test)

## Logistic Regression with GridSearchCV
param_grid_lr = {
    'C': [0.01, 0.1, 1, 10],
    'solver': ['liblinear', 'lbfgs']
}
grid_lr = GridSearchCV(LogisticRegression(max_iter=2000), param_grid_lr, cv=5)
grid_lr.fit(x_train, y_train)

y_pred_lr = grid_lr.predict(x_test)
print("\nLogistic Regression - Best Params:", grid_lr.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

## Support Vector Machine with GridSearchCV
param_grid_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}
grid_svm = GridSearchCV(SVC(), param_grid_svm, cv=5)
grid_svm.fit(x_train, y_train)

y_pred_svm = grid_svm.predict(x_test)
print("\nSVM - Best Params:", grid_svm.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))

## Random Forest with GridSearchCV
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10]
}
grid_rf = GridSearchCV(RandomForestClassifier(), param_grid_rf, cv=5)
grid_rf.fit(x_train, y_train)

y_pred_rf = grid_rf.predict(x_test)
print("\nRandom Forest - Best Params:", grid_rf.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

## Random Forest with RandomizedSearchCV
param_dist_rf = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
rand_rf = RandomizedSearchCV(RandomForestClassifier(),
                             param_dist_rf,
                             cv=5,
                             n_iter=10,
                             random_state=42)
rand_rf.fit(x_train, y_train)

y_pred_randrf = rand_rf.predict(x_test)
print("\nRandomizedSearchCV - Random Forest Best Params:", rand_rf.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred_randrf))
print(classification_report(y_test, y_pred_randrf))
#---------------------------------------------------------------------------------------------

# Step 5: Model Performance Analysis
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
print("\nStep 5: Model Performance Analysis")

## Collect predictions from Step 4 models
models = {
    "Logistic Regression": y_pred_lr,
    "SVM": y_pred_svm,
    "Random Forest (GridSearch)": y_pred_rf,
    "Random Forest (RandomizedSearch)": y_pred_randrf
}

## Store results for comparison
results = {}
for name, preds in models.items():
    results[name] = {
        "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds, average="weighted", zero_division=0),
        "Recall": recall_score(y_test, preds, average="weighted", zero_division=0),
        "F1 Score": f1_score(y_test, preds, average="weighted", zero_division=0)
    }

## Display comparison as DataFrame
results_df = pd.DataFrame(results).T
print(results_df)

## Pick the model with the highest F1 Score
best_model_name = results_df["F1 Score"].idxmax()
best_model_preds = models[best_model_name]
print(f"\nBest model based on F1 Score: {best_model_name}")

## Confusion Matrix Plot
cm = confusion_matrix(y_test, best_model_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))
disp.plot(cmap="Blues", values_format="d")
plt.title(f"Confusion Matrix - {best_model_name}")
plt.show()
#---------------------------------------------------------------------------------------------

# Step 6: Stacked Model Performance Analysis
from sklearn.ensemble import StackingClassifier
print("\nStep 6: Stacked Model Performance Analysis")

## Choosing two models from the step 4
stack_clf = StackingClassifier(
    estimators=[
        ('lr', grid_lr),         # Logistic Regression model from Step 4
        ('rf', rand_rf)          # Random Forest (best from RandomizedSearch)
    ],
    final_estimator=LogisticRegression(),   # Meta-learner
    passthrough=True,   # Pass original features to meta-learner
    cv=5
)
stack_clf.fit(x_train, y_train)
y_pred_stack = stack_clf.predict(x_test)

## Performance metrics
stack_accuracy = accuracy_score(y_test, y_pred_stack)
stack_precision = precision_score(y_test, y_pred_stack, average="weighted", zero_division=0)
stack_recall = recall_score(y_test, y_pred_stack, average="weighted", zero_division=0)
stack_f1 = f1_score(y_test, y_pred_stack, average="weighted", zero_division=0)

print(f"Stacked Model Accuracy: {stack_accuracy:.4f}")
print(f"Stacked Model Precision: {stack_precision:.4f}")
print(f"Stacked Model Recall: {stack_recall:.4f}")
print(f"Stacked Model F1 Score: {stack_f1:.4f}")

## Confusion Matrix
cm_stack = confusion_matrix(y_test, y_pred_stack)
disp_stack = ConfusionMatrixDisplay(confusion_matrix=cm_stack, display_labels=np.unique(y))
disp_stack.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix - Stacked Model")
plt.show()
#-------------------------------------------------------------------------------------------------

# Step 7: Model Evaluation
import joblib
print("\nStep 7: Model Evaluation")
selected_model = stack_clf
joblib.dump(selected_model, "best_model.joblib")
print("Model saved as 'best_model.joblib'")
loaded_model = joblib.load("best_model.joblib")

## Predic maintenance step for new coordinates
new_data = np.array([
    [9.375, 3.0625, 1.51],
    [6.995, 5.125, 0.3875],
    [0, 3.0625, 1.93],
    [9.4, 3, 1.8],
    [9.4, 3, 1.3]
])
new_data_scaled = scalar.transform(new_data)
predictions = loaded_model.predict(new_data_scaled)
print("\nPredicted Maintenance Steps for New Data:")
for coord, pred in zip(new_data, predictions):
    print(f"Coordinates {coord} -> Predicted Step: {pred}")



