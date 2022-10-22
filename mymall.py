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

st.header("My first Streamlit App")
st.write(pd.DataFrame({
    'Intplan': ['yes', 'yes', 'yes', 'no'],
    'Churn Status': [0, 0, 0, 1]
}))

# DATA EXPLORATION
m_cust = pd.read_csv('mall_customer.csv')
# m_stud = pd.read_csv('/content/student_mat.csv')

# display(m_cust)
# display(m_stud)

m_cust.head()
m_cust.tail()
m_cust.describe(include='all')
m_cust.info()

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
accuracy_score(ytest, y_model)

#from sklearn.metrics import classification_report
print(classification_report(ytest, y_model)) 

# Confusion Matrix
from sklearn.metrics import confusion_matrix 
confusion_matrix(ytest, y_model)

#Confusion Matrix
#import matplotlib.pyplot as plt
#from sklearn import metrics
#import numpy as np
confusion_matrix = metrics.confusion_matrix(ytest, y_model)

print(confusion_matrix)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix,display_labels=np.unique(y_cust))

cm_display.plot()
plt.show()

#from sklearn.metrics import classification_report
# F1 score = 2 / [ (1/precision) + (1/ recall)]
print(classification_report(ytest, y_model)) 
