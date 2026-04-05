import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# 1. Load dataset (download from Kaggle: student-mat.csv)
df = pd.read_csv('student-mat.csv')

# 2. Create target variable (Pass/Fail)
df['pass'] = df['G3'].apply(lambda x: 1 if x >= 10 else 0)

# 3. Select important features
X = df[['studytime', 'absences', 'G1', 'G2']]
y = df['pass']

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 5. Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# 6. Evaluate model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Model Accuracy:", accuracy)

# 7. Save model
pickle.dump(model, open('model.pkl', 'wb'))

print("Model trained and saved successfully!")
