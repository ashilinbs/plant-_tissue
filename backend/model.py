# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE  # Handle imbalanced data
from collections import Counter
import joblib

# Step 1: Load the dataset
data = pd.read_csv('plant_tissue_culture_dataset_5000.csv')

# Step 2: Feature Selection
features = data.drop(columns=['Regeneration_Success_Rate'])
X = features
y = data['Regeneration_Success_Rate']

# Step 3: Bin the target variable into classes
bins = [0, 25, 50, 75, 100]
labels = [0, 1, 2, 3]  # Labels for binned target classes
y_binned = pd.cut(y, bins=bins, labels=labels)

# Step 4: Preprocessing - Define the column transformer
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Step 5: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_binned, test_size=0.2, random_state=42)

# Step 6: Preprocess the training data into numeric form for SMOTE
X_train_transformed = preprocessor.fit_transform(X_train)  # Fit the preprocessor here
X_test_transformed = preprocessor.transform(X_test)  # Apply the same transformation to the test set

# Step 7: Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_transformed, y_train)

print("Class distribution after SMOTE:", Counter(y_train_resampled))

# Step 8: Create a pipeline with RandomForest
pipeline = Pipeline(steps=[
    ('classifier', RandomForestClassifier(random_state=42))
])

# Step 9: Hyperparameter Tuning with GridSearchCV
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_resampled, y_train_resampled)

# Step 10: Get the best pipeline and make predictions
best_pipeline = grid_search.best_estimator_
y_pred = best_pipeline.predict(X_test_transformed)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
print(classification_report(y_test, y_pred))

# Step 11: Visualize Original vs Predicted Rates with Scatter Plot
original_rate = y_test.astype(int)
predicted_rate = pd.Series(y_pred).astype(int)

# Create a DataFrame for comparison
comparison_df = pd.DataFrame({
    'Original Rate': original_rate,
    'Predicted Rate': predicted_rate
}).reset_index(drop=True)

# Plot: Scatter plot for better visualization
plt.figure(figsize=(12, 6))
plt.scatter(comparison_df.index, comparison_df['Original Rate'], color='blue', label='Original Rate', alpha=0.6)
plt.scatter(comparison_df.index, comparison_df['Predicted Rate'], color='orange', label='Predicted Rate', alpha=0.6)

# Titles and labels
plt.title('Comparison of Original and Predicted Regeneration Success Rate', fontsize=16)
plt.xlabel('Sample Index', fontsize=12)
plt.ylabel('Regeneration Success Rate (Binned Class)', fontsize=12)

# Show every 10th tick for readability
plt.xticks(comparison_df.index[::10])
plt.yticks(labels)

# Add legend and grid
plt.legend(loc='best')
plt.grid(True, alpha=0.3)

# Display the plot
plt.show()

# Step 12: Save the fitted preprocessor and the trained model to files
joblib.dump(preprocessor, 'fitted_preprocessor.joblib')
joblib.dump(best_pipeline, 'plant_tissue_culture_model.joblib')
print("Preprocessor and model saved successfully.")
