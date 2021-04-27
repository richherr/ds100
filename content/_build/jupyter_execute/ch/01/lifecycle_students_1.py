#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Reference: https://jupyterbook.org/interactive/hiding.html
# Use {hide, remove}-{input, output, cell} tags to hiding content

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual

sns.set()
sns.set_context('talk')
np.set_printoptions(threshold=20, precision=2, suppress=True)
pd.set_option('display.max_rows', 7)
pd.set_option('display.max_columns', 8)
pd.set_option('precision', 2)
# This option stops scientific notation for pandas
# pd.set_option('display.float_format', '{:.2f}'.format)

def display_df(df, rows=pd.options.display.max_rows,
               cols=pd.options.display.max_columns):
    with pd.option_context('display.max_rows', rows,
                           'display.max_columns', cols):
        display(df)


# # The Students of Data 100

# The data science lifecycle involves the following general steps:
# 
# 1. **Question/Problem Formulation:** 
#     1. What do we want to know or what problems are we trying to solve?  
#     1. What are our hypotheses? 
#     1. What are our metrics of success? <br/><br/>
# 1. **Data Acquisition and Cleaning:** 
#     1. What data do we have and what data do we need?  
#     1. How will we collect more data? 
#     1. How do we organize the data for analysis?  <br/><br/>
# 1. **Exploratory Data Analysis:** 
#     1. Do we already have relevant data?  
#     1. What are the biases, anomalies, or other issues with the data?  
#     1. How do we transform the data to enable effective analysis? <br/><br/>
# 1. **Prediction and Inference:** 
#     1. What does the data say about the world?  
#     1. Does it answer our questions or accurately solve the problem?  
#     1. How robust are our conclusions? <br/><br/>
#     
# We now demonstrate this process applied to a dataset of student first names from a previous offering of Data 100. In this chapter, we proceed quickly in order to give the reader a general sense of a complete iteration through the lifecycle. In later chapters, we expand on each step in this process to develop a repertoire of skills and principles.

# ## Question Formulation
# 
# We would like to figure out if the student first names give
# us additional information about the students themselves. Although this is a
# vague question to ask, it is enough to get us working with our data and we can
# make the question more precise as we go.

# ## Data Acquisition and Cleaning

# Let's begin by looking at our data, the roster of student first names that we've downloaded from a previous offering of Data 100.
# 
# Don't worry if you don't understand the code for now; we introduce the libraries in more depth soon. Instead, focus on the process and the charts that we create.

# In[2]:


import pandas as pd

students = pd.read_csv('roster.csv')
students


# We can quickly see that there are some quirks in the data. For example, one of the student's names is all uppercase letters. In addition, it is not obvious what the Role column is for.
# 
# **In Data 100, we will study how to identify anomalies in data and apply corrections.** The differences in capitalization will cause our programs to think that `'BRYAN'` and `'Bryan'` are different names when they are identical for our purposes. Let's convert all names to lower case to avoid this.

# In[3]:


students['Name'] = students['Name'].str.lower()
students


# Now that our data are in a more useful format, we proceed to exploratory data analysis.
