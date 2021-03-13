#!/usr/bin/env python
# coding: utf-8

# **The Sparks Foundation**

# **Data Science and Business Analytics Internship**

# **Task-4 :Create the Decision Tree classifier and visualize it graphically.**

# **By-Priyanka Mohanta**

# In[ ]:





# In[2]:


# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets


# In[3]:


# Loading the 
data_iris=datasets.load_iris()


# In[5]:



# Forming the iris dataframe
x=pd.DataFrame(data_iris.data, columns=data_iris.feature_names)
print(x.head(5))
y=data_iris.target
print(y)


# **Fit the classifier with default hyper-parameters**

# In[6]:


# Defining the decision tree algorithm
from sklearn.tree import DecisionTreeClassifier
DTC= DecisionTreeClassifier(random_state=1234)
model=DTC.fit(x,y)
print('Decision Tree Classifer Created')


# **Print Text Representation**

# In[7]:


from sklearn import tree
textrepre= tree.export_text(DTC)
print(textrepre)


# **Plot Tree with plot_tree**

# In[8]:


ploting= plt.figure(figsize=(25,20))
tree_graph= tree.plot_tree(DTC,feature_names=data_iris.feature_names, 
                          class_names=data_iris.target_names,filled=True)


# **THANK YOU**

# In[ ]:




