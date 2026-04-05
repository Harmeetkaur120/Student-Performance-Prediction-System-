import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle

# Sample dataset
data = {
    'hours': [1,2,3,4,5,6,7,8],
    'attendance': [50,60,65,70,75,80,85,90],
    'pass': [0,0,0,1,1,1,1,1]
}

df = pd.DataFrame(data)

X = df[['hours','attendance']]
y = df['pass']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open('model.pkl', 'wb'))

print("Model trained and saved!")
