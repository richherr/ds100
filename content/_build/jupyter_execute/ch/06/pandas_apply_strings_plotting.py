#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

sns.set()
sns.set_context('talk')
np.set_printoptions(threshold=20, precision=2, suppress=True)
pd.options.display.max_rows = 7
pd.options.display.max_columns = 8
pd.set_option('precision', 2)
# This option stops scientific notation for pandas
# pd.set_option('display.float_format', '{:.2f}'.format)


# # Apply, Strings, and Plotting
# 
# In this section, we will answer the question:
# 
# **Can we use the last letter of a name to predict the sex of the baby?**
# 
# Here's the Baby Names dataset once again:

# In[3]:


baby = pd.read_csv('babynames.csv')
baby.head()
# the .head() method outputs the first five rows of the DataFrame


# **Breaking the Problem Down**
# 
# Although there are many ways to see whether prediction is possible, we will use plotting in this section. We can decompose this question into two steps:
# 
# 1. Compute the last letter of each name.
# 1. Group by the last letter and sex, aggregating on Count.
# 1. Plot the counts for each sex and letter.

# ## Apply
# 
# `pandas` Series contain an `.apply()` method that takes in a function and applies it to each value in the Series.

# In[7]:


names = baby['Name']
names.apply(len)


# To extract the last letter of each name, we can define our own function to pass into `.apply()`:

# In[8]:


def last_letter(string):
    return string[-1]

names.apply(last_letter)


# ## String Manipulation
# 
# Although `.apply()` is flexible, it is often faster to use the built-in string manipulation functions in `pandas` when dealing with text data.
# 
# `pandas` provides access to string manipulation functions using the `.str` attribute of Series.

# In[9]:


names = baby['Name']
names.str.len()


# We can directly slice out the last letter of each name in a similar way.

# In[10]:


names.str[-1]


# We suggest looking at the docs for the full list of string methods ([link](https://pandas.pydata.org/pandas-docs/stable/text.html)).
# 
# We can now add this column of last letters to our `baby` DataFrame.

# In[11]:


baby['Last'] = names.str[-1]
baby


# ## Grouping
# 
# To compute the sex distribution for each last letter, we need to group by both Last and Sex.

# In[13]:


# Shorthand for baby.groupby(['Last', 'Sex']).agg(np.sum)
baby.groupby(['Last', 'Sex']).sum()


# Notice that `Year` is also summed up since each non-grouped column is passed into the aggregation function. To avoid this, we can select out the desired columns before calling `.groupby()`.

# In[15]:


# When lines get long, you can wrap the entire expression in parentheses
# and insert newlines before each method call
letter_dist = (
    baby[['Last', 'Sex', 'Count']]
    .groupby(['Last', 'Sex'])
    .sum()
)
letter_dist


# ## Plotting
# 
# `pandas` provides built-in plotting functionality for most basic plots, including bar charts, histograms, line charts, and scatterplots. To make a plot from a DataFrame, use the `.plot` attribute:

# In[26]:


# We use the figsize option to make the plot larger
letter_dist.plot.barh(figsize=(10, 10))


# Although this plot shows the distribution of letters and sexes, the male and female bars are difficult to tell apart. By looking at the `pandas` docs on plotting ([link](https://pandas.pydata.org/pandas-docs/stable/visualization.html)) we learn that `pandas` plots one group of bars for row column in the DataFrame, showing one differently colored bar for each column. This means that a pivoted version of the `letter_dist` table will have the right format.

# In[20]:


letter_pivot = pd.pivot_table(
    baby, index='Last', columns='Sex', values='Count', aggfunc='sum'
)
letter_pivot


# In[27]:


letter_pivot.plot.barh(figsize=(10, 10))


# Notice that `pandas` conveniently generates a legend for us as well. However, this is still difficult to interpret. We plot the counts for each letter and sex which causes some bars to appear very long and others to be almost invisible. We should instead plot the proportion of male and female babies within each last letter.

# In[32]:


total_for_each_letter = letter_pivot['F'] + letter_pivot['M']

letter_pivot['F prop'] = letter_pivot['F'] / total_for_each_letter
letter_pivot['M prop'] = letter_pivot['M'] / total_for_each_letter
letter_pivot


# In[34]:


(letter_pivot[['F prop', 'M prop']]
 .sort_values('M prop') # Sorting orders the plotted bars
 .plot.barh(figsize=(10, 10))
)


# ## Summary
# 
# We can see that almost all first names that end in 'p' are male and names that end in 'a' are female! In general, the difference between bar lengths for many letters implies that we can often make a good guess to a person's sex if we just know the last letter of their first name.
# 
# We've learned to express the following operations in `pandas`:
# 
# | Operation | `pandas` |
# | --------- | -------  |
# | Applying a function elementwise | `series.apply(func)` |
# | String manipulation | `series.str.func()` |
# | Plotting | `df.plot.func()` |
