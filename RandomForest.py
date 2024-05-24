import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
data = pd.read_csv("D:\\heart disease\\dataset_heart.csv")


# Handle missing values
data = data.dropna()

# Check dataset size
if len(data) == 0:
    raise ValueError("Empty dataset after preprocessing. Check data preprocessing steps.")

# Split features and target variable
X = data.drop(columns=['heart disease'])
y = data['heart disease']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check if train set is empty
if len(X_train) == 0:
    raise ValueError("Empty train set. Adjust test_size parameter or acquire more data.")

# Instantiate the Random Forest classifier
rf_classifier = RandomForestClassifier()

# Train the classifier
rf_classifier.fit(X_train, y_train)

# Make predictions
y_pred = rf_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy Of Random Forest: ", accuracy)


import pickle

# Save the trained model to a .pkl file
with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(rf_classifier, f)