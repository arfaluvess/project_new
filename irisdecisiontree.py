from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.tree import plot_tree

Xtrain, Xtest, ytrain, ytest = train_test_split(X_iris, y_iris,random_state=1)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(Xtrain, ytrain)

clf.fit(Xtrain, ytrain)   

clf.score(Xtest, ytest)