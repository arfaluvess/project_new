from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.tree import plot_tree

iris = load_iris()
(X_iris, y_iris) = load_iris(return_X_y = True)
# X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris, random_state = 40)
Xtrain, Xtest, ytrain, ytest = train_test_split(X_iris, y_iris,random_state=1)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(Xtrain, ytrain)

clf.fit(Xtrain, ytrain)   

clf.score(Xtest, ytest)
