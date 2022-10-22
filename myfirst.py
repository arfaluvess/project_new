import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.header("My first Streamlit App")

iris = sns.load_dataset('iris')
fig = plt.figure(figsize=(10, 4))
sns.boxplot(data=iris)
st.pyplot(fig)
