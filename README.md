# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the Iris dataset and split it into training and testing sets.

2. Create an SGDClassifier model with suitable hyperparameters.

3. Train the model using the training data.

4. Predict the species for test data and evaluate accuracy.


## Program:
```
Program to implement the prediction of iris species using SGD Classifier.
Developed by: ASHIKA TR
RegisterNumber:  212224220011

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))


```

## Output:
<img width="1916" height="950" alt="image" src="https://github.com/user-attachments/assets/eb83d600-e055-4b96-b418-bff4a40c2344" />



## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
