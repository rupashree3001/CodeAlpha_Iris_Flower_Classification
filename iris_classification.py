import pandas as pd

# Load the Iris dataset
data = pd.read_csv("Iris.csv")

# Display the first 5 rows
print("First 5 rows of the dataset:")
print(data.head())

# Display the dataset information
print("\nDataset Information:")
print(data.info())

# Display the number of rows and columns
print("\nDataset Shape:")
print(data.shape)
# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Check for duplicate rows
print("\nNumber of Duplicate Rows:")
print(data.duplicated().sum())

# Display statistical summary
print("\nStatistical Summary:")
print(data.describe())

# Display the number of samples in each species
print("\nSpecies Distribution:")
print(data["Species"].value_counts())
# Separate features and target
X = data[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
y = data["Species"]

print("\nFeatures:")
print(X.head())

print("\nTarget:")
print(y.head())
# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTraining data shape:")
print(X_train.shape)

print("\nTesting data shape:")
print(X_test.shape)
# Train the K-Nearest Neighbors (KNN) model
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=5)

model.fit(X_train, y_train)

print("\nKNN model trained successfully!")
# Make predictions on the test data
y_pred = model.predict(X_test)

print("\nPredicted values:")
print(y_pred)
# Evaluate the model accuracy
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(accuracy)
print(f"Accuracy Percentage: {accuracy * 100:.2f}%")
# Classification report
from sklearn.metrics import classification_report

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
# Confusion Matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Display the confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=model.classes_,
    yticklabels=model.classes_
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("KNN Confusion Matrix")
plt.show()