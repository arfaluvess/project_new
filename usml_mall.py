# -*- coding: utf-8 -*-
"""project_usml.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ZH4Gn3FEG2Zwc2Pg5l5zs7rrpjSyRJ1b
"""

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans 

mall_df = pd.read_csv('mall_customer.csv')
st.write(mall_df.head())


X = mall_df[['Annual_Income_(k$)','Spending_Score']]
st.write(X)

# Scatterplot of the input data
fig=plt.figure(figsize=(10, 6))
sns.scatterplot(x='Annual_Income_(k$)', y='Spending_Score', data=X)
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)') 
plt.title('Spending Score (1-100) vs Annual Income (k$)')
st.pyplot(fig)

# Find the KMeans of Mall Customer
wcss=[]
for i in range(1,11):
    km=KMeans(n_clusters=i)
    km.fit(X)
    wcss.append(km.inertia_)
    print(wcss)

#The elbow curve
plt.figure(figsize=(12,6))
plt.plot(range(1,11),wcss)
plt.plot(range(1,11),wcss, linewidth=2, color="red", marker ="8")
plt.xlabel("K Value")
plt.xticks(np.arange(1,11,1))
plt.ylabel("WCSS")
plt.show()

#Taking 5 clusters
km1=KMeans(n_clusters=5)
#Fitting the input data
km1.fit(X)
#predicting the labels of the input data
y=km1.predict(X)
#adding the labels to a column named label
mall_df["label"] = y
#The new dataframe with the clustering done
mall_df.head()

#Scatterplot of the clusters
plt.figure(figsize=(10,6))
sns.scatterplot(x = 'Annual_Income_(k$)',y = 'Spending_Score',hue="label", palette=['green','orange','brown','dodgerblue','red'], legend='full',data = mall_df  ,s = 60 )
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)') 
plt.title('Spending Score (1-100) vs Annual Income (k$)')
plt.show()
