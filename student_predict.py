import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the student data from CSV file
data = pd.read_csv('student_data.csv')

# We don't need the 'Name' column for predictions, so let's drop it
new_data = data.drop('Name', axis=1)

# Convert gender into numbers: Male = 0, Female = 1
new_data['Gender'] = new_data['Gender'].map({'Male': 0, 'Female': 1})

# The 'Passed' column has words like 'Yes' or 'No' — convert these to 1 and 0 for our model
le = LabelEncoder()
new_data['Passed'] = le.fit_transform(new_data['Passed'])

# Pick the columns that will help predict if a student passed
X = new_data[['Gender', 'Age', 'StudyHoursPerWeek', 'AttendancePercent', 'PreviousGrade', 'FinalGrade']]

# This is what we want to predict: whether the student passed (1) or not (0)
y = new_data['Passed']

# Split the data: 80% for training, 20% for testing — helps us check how well our model works
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features so they all have the same range (important for some ML models)
scaler = StandardScaler()

# Fit the scaler to training data and transform it
X_train_scaled = scaler.fit_transform(X_train)

# Use the same scaling on the test data (don’t fit again!)
X_test_scaled = scaler.transform(X_test)

# Turn scaled data back into DataFrames to keep track of column names easily
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

# Now, train a simple linear regression model using the scaled training data
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predict 'Passed' values on the test set — these will be numbers, not just 0 or 1
predictions = model.predict(X_test_scaled)

# Convert predictions to 0 or 1 by using 0.5 as a cutoff point
predicted_classes = []
for p in predictions:
    predicted_classes.append(1 if p > 0.5 else 0)

# Let's focus on students aged 16 now
predict_passed = new_data[new_data['Age'] == 16]

# Get their feature data to predict if they passed
input_predict_passed = predict_passed[['Gender', 'Age', 'StudyHoursPerWeek', 'AttendancePercent', 'PreviousGrade', 'FinalGrade']]

# Scale this data using our scaler (same as before!)
input_predict_passed_scaled = scaler.transform(input_predict_passed)

# Put the scaled data into a DataFrame just to keep things neat
input_predict_passed_scaled_df = pd.DataFrame(input_predict_passed_scaled, columns=input_predict_passed.columns)

# Predict passing chances for these 16-year-olds
final_predict_grade = model.predict(input_predict_passed_scaled_df)

# Show what actually happened with final grades and passing for age 16 students
print("Actual grade and pass status for students aged 16:")
print(predict_passed[['FinalGrade', "Passed"]])

# Show what our model predicts for those students
print("Predicted passing chances for students aged 16:")
print(final_predict_grade)
