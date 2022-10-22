import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns

st.header("My first Streamlit App")
st.write(pd.DataFrame({
    'Intplan': ['yes', 'yes', 'yes', 'no'],
    'Churn Status': [0, 0, 0, 1]
}))

iris = pd.read_csv('https://github.com/arfaluvess/project_new/blob/d1c562635e29fe6a40ca028b12f11781d8b21e00/iris.csv')
sns.boxplot(data=iris)
