import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the datasets
gender_submission = pd.read_csv('data/gender_submission.csv')
test = pd.read_csv('data/test.csv')
train = pd.read_csv('data/train.csv')

# Display the first few rows of each dataset to understand their structure
gender_submission_head = gender_submission.head()
test_head = test.head()
train_head = train.head()

gender_submission_info = gender_submission.info()
test_info = test.info()
train_info = train.info()

(gender_submission_head, test_head, train_head)

# Find survival rate
survival_rate = train['Survived'].mean()
print(f"Survival Rate: {survival_rate:.2f}")

# Plot survivors count by gender
sns.countplot(x='Sex', hue='Survived', data=train)
plt.title('Survival Count by Gender')
plt.show()

# Age distribution of passengers
sns.histplot(train['Age'].dropna(), kde=True)
plt.title('Age Distribution of Passengers')
plt.show()

# Plot survivors count by passenger class
sns.countplot(x='Pclass', hue='Survived', data=train)
plt.title('Survival Count by Passenger Class')
plt.show()

# Missing values in dataset
missing_values_train = train.isnull().sum()
missing_values_test = test.isnull().sum()
print("Missing Values in Train Dataset:\n", missing_values_train)
print("Missing Values in Test Dataset:\n", missing_values_test)