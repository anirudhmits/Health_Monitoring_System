import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

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

# Scale the input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the scaled data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Check if train set is empty
if len(X_train) == 0:
    raise ValueError("Empty train set. Adjust test_size parameter or acquire more data.")

# Instantiate the KNN classifier
knn_classifier = KNeighborsClassifier()

# Train the classifier
knn_classifier.fit(X_train, y_train)

# Make predictions
y_pred = knn_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy Of KNN: ", accuracy)

import pickle
# Save the trained model to a .pkl file
with open('knn_model.pkl', 'wb') as f:
    pickle.dump(knn_classifier, f)
