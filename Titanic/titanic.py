import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("data/train.csv")

y = data["Survived"]
X = data.drop(["Survived"], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

#  Looking at the data a bit

X_train.shape

X_train.head()

X_train.columns


X_train["PassengerId"]

X_train.describe()

X_train["Age"].hist(bins=30)
plt.show()

X_train["Fare"].hist(bins = 30)
plt.show()


#  First model
features_dummy = ["Pclass", "Sex", "SibSp", "Parch"]
X_train_dummy = pd.get_dummies(X_train[features_dummy])
X_test_dummy = pd.get_dummies(X_test[features_dummy])

model = RandomForestClassifier(n_estimators = 100, max_depth = 5, random_state = 42)
model.fit(X_test_dummy, y_test)

y_pred = model.predict(X_test_dummy)

output = pd.DataFrame({'PassengerId': X_test.PassengerId, 'Survived': y_pred})

output.head()

accuracy_score(y_pred, y_test)


