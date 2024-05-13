import joblib
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X, Y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, random_state=20)

dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
y_predict = dtc.predict(X_test)
print(f"mean_squared_error: {mean_squared_error(y_test, y_predict)}")
print(f"accuracy_score: {accuracy_score(y_test, y_predict)}")
print(f"r2_score: {r2_score(y_test, y_predict)}")
joblib.dump(dtc, "dtc.model")
