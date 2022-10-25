# Using Iris dataset 
import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.tree import plot_tree
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import cross_val_score

# Title and Subheader
st.title("Iris EDA App")
st.subheader("EDA Web App with Streamlit ")

# PROBABILITY USING SVM 
st.header('PREPARATION OF DATA')
# SVM DATASET
iris = sns.load_dataset('iris') # returns a pandas dataframe
X_iris = iris.drop('species', axis=1)  
X_iris
y_iris = iris['species']
y_iris

# iris = load_iris()
# (X_iris, y_iris) = load_iris(return_X_y = True)
X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris, random_state = 0)


clf = SVC(kernel='rbf', C=1).fit(X_train, y_train)
st.write('Iris Flower Dataset')
st.write('Accuracy of RBF SVC classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
st.write('Accuracy of RBF SVC classifier on test set: {:.2f}'
     .format(clf.score(X_test, y_test)))

st.header('INFORMATION ON USING NAIVE BAYES)
# from sklearn.naive_bayes import GaussianNB # 1. choose model class
model = GaussianNB()                       # 2. instantiate model
model.fit(X_train, y_train)                  # 3. fit model to data
y_model = model.predict(X_test)             # 4. predict on new data
y_model

# Accuracy score
st.write(accuracy_score(y_test, y_model))

# Classification report finding
st.write(classification_report(y_test, y_model)) 

# Confusion Matrix
confusion_matrix(y_test, y_model)

#Confusion Matrix
n_cm = metrics.confusion_matrix(y_test, y_model)
st.write(n_cm)

# cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = n_cm,display_labels=np.unique(y_iris))
# # fig, ax = plt.subplots()
# # cm_display.plot()
# # st.pyplot(fig)

# fig, ax = plt.subplots(figsize=(10,10))
# st.write(sns.heatmap(iris.corr(), annot=True,linewidths=0.5))
# cm_display.plot()
# st.pyplot(fig)

# # F1 score = 2 / [ (1/precision) + (1/ recall)]
# st.write(classification_report(y_test, y_model)) 

st.header("INFORMATION ON PROBABILITY USING SVM")

# SVM METHOD 
classifier = SVC(kernel = 'linear', random_state = 0)

#Fit the model for the data
classifier.fit(X_train, y_train)

# FIND THE ACCURACY
#Make the prediction
y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
st.write(cm)

# Check for the accuracy
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
st.write("Accuracy: {:.2f} %".format(accuracies.mean()*100))
st.write("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

# Show the plotting visualization
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm,display_labels=np.unique(y_iris))
# cm_display.plot()
# plt.show()

fig, ax = plt.subplots(figsize=(10,10))
st.write(sns.heatmap(cm, annot=True,linewidths=0.5))
cm_display.plot()
st.pyplot(fig)

# F1 score = 2 / [ (1/precision) + (1/ recall)]
st.write(classification_report(y_test, y_pred)) 

st.header("INFORMATION ON DECISION TREE")

# DECISION TREE
Xtrain, Xtest, ytrain, ytest = train_test_split(X_iris, y_iris,random_state=1)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(Xtrain, ytrain)

st.write(clf.fit(Xtrain, ytrain))   
clf.score(Xtest, ytest)

fig=plt.figure(figsize=(15,8))
tree.plot_tree(clf.fit(Xtrain, ytrain))
st.pyplot(fig)
