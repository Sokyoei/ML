import lightgbm
import joblib
from sklearn.datasets import load_wine
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score
from sklearn.model_selection import train_test_split

X, Y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, random_state=20)

lgbm = lightgbm.LGBMClassifier()
lgbm.fit(X_train, y_train)
y_predict = lgbm.predict(X_test)
print(f"mean_squared_error: {mean_squared_error(y_test, y_predict)}")
print(f"accuracy_score: {accuracy_score(y_test, y_predict)}")
print(f"r2_score: {r2_score(y_test, y_predict)}")
joblib.dump(lgbm, "lgbm.model")
