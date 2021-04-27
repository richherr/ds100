#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import warnings
# Ignore numpy dtype warnings. These warnings are caused by an interaction
# between numpy and Cython and can be safely ignored.
# Reference: https://stackoverflow.com/a/40846742
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual
import nbinteract as nbi

sns.set()
sns.set_context('talk')
np.set_printoptions(threshold=20, precision=2, suppress=True)
pd.options.display.max_rows = 7
pd.options.display.max_columns = 8
pd.set_option('precision', 2)
# This option stops scientific notation for pandas
# pd.set_option('display.float_format', '{:.2f}'.format)


# # Indexes, Slicing, and Sorting
# 
# In the remaining sections of this chapter we will work with the Baby Names dataset from Chapter 1. We will pose a question, break the question down into high-level steps, then translate each step into Python code using `pandas` DataFrames. We begin by importing `pandas`:

# In[1]:


import pandas as pd


# Now we can read in the data using `pd.read_csv` ([docs](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html)).

# In[9]:


baby = pd.read_csv('babynames.csv')
baby


# Note that for the code above to work, the `babynames.csv` file must be located in the same directory as this notebook. We can check what files are in the current folder by running the `ls` command-line tool:

# In[3]:


get_ipython().system('ls')


# When we use `pandas` to read in data, we get a DataFrame. A DataFrame is a tabular data structure where each column is labeled (in this case 'Name', 'Sex', 'Count', 'Year') and each row is labeled (in this case 0, 1, 2, ..., 1891893). Note that the Table object introduced in Data 8 only labels columns while DataFrames label both columns and rows.
# 
# The labels of a DataFrame are called the *indexes* of the DataFrame and make many data manipulations easier.

# ## Slicing
# 
# Let's use `pandas` to answer the following question:
# 
# **What were the five most popular baby names in 2016?**
# 
# ### Breaking the Problem Down
# 
# We can decompose this question into the following simpler table manipulations:
# 
# 1. Slice out the rows for the year 2016.
# 2. Sort the rows in descending order by Count.
# 
# Now, we can express these steps in `pandas`.
# 
# ### Slicing using `.loc`
# 
# To select subsets of a DataFrame, we use the `.loc` slicing syntax. The first argument is the label of the row and the second is the label of the column:

# In[10]:


baby


# In[11]:


baby.loc[1, 'Name'] # Row labeled 1, Column labeled 'Name'


# To slice out multiple rows or columns, we can use `:`. Note that `.loc` slicing is inclusive, unlike Python's slicing.

# In[12]:


# Get rows 1 through 5, columns Name through Count inclusive
baby.loc[1:5, 'Name':'Count']


# We will often want a single column from a DataFrame:

# In[13]:


baby.loc[:, 'Year']


# Note that when we select a single column, we get a `pandas` Series. A Series is like a one-dimensional NumPy array since we can perform arithmetic on all the elements at once.

# In[14]:


baby.loc[:, 'Year'] * 2


# To select out specific columns, we can pass a list into the `.loc` slice:

# In[15]:


# This is a DataFrame again
baby.loc[:, ['Name', 'Year']]


# Selecting columns is common, so there's a shorthand.

# In[16]:


# Shorthand for baby.loc[:, 'Name']
baby['Name']


# In[17]:


# Shorthand for baby.loc[:, ['Name', 'Count']]
baby[['Name', 'Count']]


# #### Slicing rows using a predicate
# 
# To slice out the rows with year 2016, we will first create a Series containing `True` for each row we want to keep and `False` for each row we want to drop. This is simple because math and boolean operators on Series are applied to each element in the Series.

# In[18]:


# Series of years
baby['Year']


# In[21]:


# Compare each year with 2016
baby['Year'] == 2016


# Once we have this Series of `True` and `False`, we can pass it into `.loc`.

# In[23]:


# We are slicing rows, so the boolean Series goes in the first
# argument to .loc
baby_2016 = baby.loc[baby['Year'] == 2016, :]
baby_2016


# ## Sorting Rows
# 
# The next step is the sort the rows in descending order by 'Count'. We can use the `sort_values()` function.

# In[25]:


sorted_2016 = baby_2016.sort_values('Count', ascending=False)
sorted_2016


# Finally, we will use `.iloc` to slice out the first five rows of the DataFrame. `.iloc` works like `.loc` but takes in numerical indices instead of labels. It does not include the right endpoint in its slices, like Python's list slicing.

# In[27]:


# Get the value in the zeroth row, zeroth column
sorted_2016.iloc[0, 0]


# In[28]:


# Get the first five rows
sorted_2016.iloc[0:5]


# ## Summary
# 
# We now have the five most popular baby names in 2016 and learned to express the following operations in `pandas`:
# 
# | Operation | `pandas` |
# | --------- | -------  |
# | Read a CSV file | `pd.read_csv()` |
# | Slicing using labels or indices | `.loc` and `.iloc` |
# | Slicing rows using a predicate | Use a boolean-valued Series in `.loc` |
# | Sorting rows | `.sort_values()` |
