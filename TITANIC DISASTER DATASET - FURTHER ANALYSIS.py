import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,ConfusionMatrixDisplay

# Load datasets
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
gender_submission = pd.read_csv('data/gender_submission.csv')

# Display the first few rows of each dataset
gender_submission.head(), test.head(), train.head()

# Display basic statistics for the train dataset
train_summary = train.describe(include='all')

# Checking for missing values in the train dataset
missing_values_train = train.isnull().sum()

# Visualizing the distribution of numerical features
numerical_features = ['Age', 'Fare']
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
for i, feature in enumerate(numerical_features):
    sns.histplot(train[feature].dropna(), kde=True, ax=axes[i])
    axes[i].set_title(f'Distribution of {feature}')
    
# Visualizing the distribution of categorical features
categorical_features = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'Survived']
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for i, feature in enumerate(categorical_features):
    sns.countplot(x=train[feature], ax=axes[i])
    axes[i].set_title(f'Countplot of {feature}')

plt.tight_layout()
plt.show()

# Display summary statistics and missing values
train_summary, missing_values_train

# Handling missing values

# Fill missing 'Age' values with the median age
train['Age'].fillna(train['Age'].median(), inplace=True)

# Fill missing 'Embarked' values with the most frequent value
train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)

# Dropping 'Cabin' column due to a high number of missing values
train.drop(columns=['Cabin'], inplace=True)

# Verify that there are no more missing values
missing_values_after = train.isnull().sum()

missing_values_after

# Assuming 'train' is a DataFrame that has been previously defined
# Dropping non-numeric columns for correlation analysis
numeric_train = train.select_dtypes(include=[float, int])

# Correlation analysis
correlation_matrix = numeric_train.corr()

# Plotting the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# Display the correlation of features with the target variable 'Survived'
if 'Survived' in correlation_matrix.columns:
    correlation_with_survived = correlation_matrix['Survived'].sort_values(ascending=False)
    print(correlation_with_survived)
else:
    print("The target variable 'Survived' is not in the correlation matrix.")

# Encode categorical variables using one-hot encoding
train_encoded = pd.get_dummies(train, columns=['Sex', 'Embarked'], drop_first=True)

# Display the first few rows of the encoded dataset
train_encoded.head()

# Drop unnecessary columns
X = train_encoded.drop(columns=['PassengerId', 'Name', 'Ticket', 'Survived'])
y = train_encoded['Survived']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the logistic regression model
logreg = LogisticRegression(max_iter=200)
logreg.fit(X_train, y_train)

# Predict on the validation set
y_pred = logreg.predict(X_val)

# Evaluate the model
accuracy = accuracy_score(y_val, y_pred)
conf_matrix = confusion_matrix(y_val, y_pred)
class_report = classification_report(y_val, y_pred)

print(accuracy)
print("\n")
print(class_report)

# Visualize confusion matrix
style.use('classic')
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=logreg.classes_)
disp.plot()

