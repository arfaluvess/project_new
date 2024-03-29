# IMPORT LIBRARY
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix 

st.header("Machine Learning Application")
# st.write(pd.DataFrame({
#     'Intplan': ['yes', 'yes', 'yes', 'no'],
#     'Churn Status': [0, 0, 0, 1]
# }))

# DATA EXPLORATION
m_cust = pd.read_csv('mall_customer.csv')
# m_stud = pd.read_csv('/content/student_mat.csv')

# display(m_cust)
# display(m_stud)

st.write(m_cust.head())
st.write(m_cust.tail())
# st.write(m_cust.describe(include='all'))
# st.write(m_cust.info())

X_cust = m_cust.drop(['CustomerID','Genre'], axis=1)  
X_cust
y_cust = m_cust['Genre']
y_cust

#from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X_cust, y_cust)

#from sklearn.naive_bayes import GaussianNB # 1. choose model class
model = GaussianNB()                       # 2. instantiate model
model.fit(Xtrain, ytrain)                  # 3. fit model to data
y_model = model.predict(Xtest)             # 4. predict on new data
y_model

#from sklearn.metrics import accuracy_score
acc=accuracy_score(ytest, y_model)
st.write(acc)

print(classification_report(ytest, y_model)) 

# Confusion Matrix
confusion_matrix(ytest, y_model)

#Confusion Matrix
confusion_matrix = metrics.confusion_matrix(ytest, y_model)
st.write(confusion_matrix)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix,display_labels=np.unique(y_cust))

fig, ax = plt.subplots(figsize=(10,10))
st.write(sns.heatmap(confusion_matrix, annot=True,linewidths=0.5))
cm_display.plot()
st.pyplot(fig)

# F1 score = 2 / [ (1/precision) + (1/ recall)]
st.write(classification_report(ytest, y_model)) 
