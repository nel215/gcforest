# coding:utf-8
from sklearn.datasets import load_digits
from gc_forest import GCForest

X, y = load_digits(return_X_y=True)
n = X.shape[0]
X = X.reshape((n, 8, 8))
model = GCForest()
model.fit(X, y)
