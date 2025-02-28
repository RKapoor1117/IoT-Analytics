#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Data Handling
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Data Preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Machine Learning Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[3]:


# Load dataset
file_path = "ai4i2020.csv"  # Change this if your file has a different name
df = pd.read_csv(file_path)

# Display first few rows
df.head()


# In[5]:


# Check basic dataset information
print(df.info())


# In[6]:


# Check for missing values
print("\nMissing values:\n", df.isnull().sum())


# In[7]:


# Display summary statistics
print("\nSummary statistics:\n", df.describe())


# In[9]:


# Encode categorical columns (if any)
le = LabelEncoder()
df["Type"] = le.fit_transform(df["Type"])  # Encoding machine types

# Standardize numerical sensor data
scaler = StandardScaler()
num_features = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
df[num_features] = scaler.fit_transform(df[num_features])

# Display processed dataset
df.head()


# In[12]:


df = df.drop(df.columns[[0, 1]], axis=1)


# In[13]:


df.head()


# ## EDA

# ### Sensor Reading Distribution

# In[10]:


plt.figure(figsize=(12, 6))
sns.histplot(df["Rotational speed [rpm]"], bins=50, kde=True)
plt.title("Distribution of Rotational Speed")
plt.xlabel("RPM")
plt.ylabel("Count")
plt.show()


# ### Correlation Matrix for IoT Sensors

# In[16]:


plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", annot_kws={"size": 10})  # Adjust size
plt.title("Feature Correlation Matrix")
plt.show()


# ### Failure Distribution

# In[19]:


plt.figure(figsize=(10,6))
ax = sns.countplot(x=df["Machine failure"])
plt.title("Failure Occurrences in Machines")
plt.xlabel("Failure (0 = No, 1 = Yes)")
plt.ylabel("Count")

# Add labels on top of bars
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', 
                (p.get_x() + p.get_width() / 2, p.get_height()), 
                ha='center', va='bottom', fontsize=12, color='black')

plt.show()


# ### Train-Test Splitting 

# In[20]:


# Define features and target
X = df.drop(columns=["Machine failure"])  # All features except target
y = df["Machine failure"]  # Target column

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training Set Size: {X_train.shape[0]} samples")
print(f"Test Set Size: {X_test.shape[0]} samples")


# In[25]:


print(list(X))


# In[30]:


# Train a Random Forest model for IoT failure prediction
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)


# ### Balancing the Dataset

# In[39]:


from imblearn.over_sampling import SMOTE


# In[40]:


# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Check class distribution after balancing
print("Balanced Class Distribution:\n", pd.Series(y_train_balanced).value_counts())


# In[41]:


def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_name="Model"):
    """
    Trains a model and evaluates performance.
    """
    # Train Model
    model.fit(X_train, y_train)
    
    # Make Predictions
    y_pred = model.predict(X_test)

    # Evaluate Model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n🔹 {model_name} Accuracy: {accuracy:.2f}")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    plt.figure(figsize=(6,4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", 
                xticklabels=["No Failure", "Failure"], yticklabels=["No Failure", "Failure"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{model_name} - Confusion Matrix")
    plt.show()


# In[42]:


print("🔹 Training Models on Unbalanced Data")

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
train_and_evaluate_model(rf, X_train, y_train, X_test, y_test, "Random Forest (Unbalanced)")


# In[43]:


print("\n🔹 Training Models on Balanced Data")

# Random Forest
train_and_evaluate_model(rf, X_train_balanced, y_train_balanced, X_test, y_test, "Random Forest (Balanced)")


# In[44]:


# Feature Importance
importance = model.feature_importances_
feature_importance = pd.DataFrame({"Feature": X_train.columns, "Importance": importance})
feature_importance = feature_importance.sort_values(by="Importance", ascending=False)

# Plot Feature Importance with Data Labels
plt.figure(figsize=(14,6))
ax = sns.barplot(x="Importance", y="Feature", data=feature_importance, palette="viridis")
plt.title("Feature Importance in Predicting Machine Failure")

# Add data labels on bars
for p in ax.patches:
    ax.annotate(f"{p.get_width():.4f}", (p.get_width(), p.get_y() + p.get_height()/2),
                ha="left", va="center", fontsize=10, color="black")

plt.show()


import pickle

# Save the trained Random Forest model (Unbalanced)
with open("rf_model.pkl", "wb") as file:
    pickle.dump(rf, file)

print("✅ Random Forest model (Unbalanced) saved as 'rf_model.pkl'!")




