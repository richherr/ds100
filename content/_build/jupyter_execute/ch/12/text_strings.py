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


# # Python String Methods
# 
# Python provides a variety of methods for basic string manipulation. Although simple, these methods form the primitives that piece together to form more complex string operations. We will introduce Python's string methods in the context of a common use case for working with text: data cleaning.
# 
# ## Cleaning Text Data
# 
# Data often comes from several different sources that each implements its own way of encoding information. In the following example, we have one table that records the state that a county belongs to and another that records the population of the county.

# In[2]:


state = pd.DataFrame({
    'County': [
        'De Witt County',
        'Lac qui Parle County',
        'Lewis and Clark County',
        'St John the Baptist Parish',
    ],
    'State': [
        'IL',
        'MN',
        'MT',
        'LA',
    ]
})
population = pd.DataFrame({
    'County': [
        'DeWitt  ',
        'Lac Qui Parle',
        'Lewis & Clark',
        'St. John the Baptist',
    ],
    'Population': [
        '16,798',
        '8,067',
        '55,716',
        '43,044',
    ]
})


# In[3]:


state


# In[4]:


population


# We would naturally like to join the `state` and `population` tables using the `County` column. Unfortunately, not a single county is spelled the same in the two tables. This example is illustrative of the following common issues in text data:
# 
# 1.  Capitalization: `qui` vs `Qui`
# 1.  Different punctuation conventions: `St.` vs `St` 
# 1.  Omission of words: `County`/`Parish` is absent in the `population` table
# 1.  Use of whitespace: `DeWitt` vs `De Witt`
# 1.  Different abbreviation conventions: `&` vs `and`

# ## String Methods
# 
# Python's string methods allow us to start resolving these issues. These methods are conveniently defined on all Python strings and thus do not require importing other modules. Although it is worth familiarizing yourself with [the complete list of string methods](https://docs.python.org/3/library/stdtypes.html#string-methods), we describe a few of the most commonly used methods in the table below.

# | Method              | Description                                                                 |
# | ------------------- | --------------------------------------------------------------------------- |
# | `str[x:y]`          | Slices `str`, returning indices x (inclusive) to y (not inclusive)          |
# | `str.lower()`       | Returns a copy of a string with all letters converted to lowercase          |
# | `str.replace(a, b)` | Replaces all instances of the substring `a` in `str` with the substring `b` |
# | `str.split(a)`      | Returns substrings of `str` split at a substring `a`                        |
# | `str.strip()`       | Removes leading and trailing whitespace from `str`                          |
# 

# We select the string for St. John the Baptist parish from the `state` and `population` tables and apply string methods to remove capitalization, punctuation, and `county`/`parish` occurrences.

# In[5]:


john1 = state.loc[3, 'County']
john2 = population.loc[3, 'County']

(john1
 .lower()
 .strip()
 .replace(' parish', '')
 .replace(' county', '')
 .replace('&', 'and')
 .replace('.', '')
 .replace(' ', '')
)


# Applying the same set of methods to `john2` allows us to verify that the two strings are now identical.

# In[6]:


(john2
 .lower()
 .strip()
 .replace(' parish', '')
 .replace(' county', '')
 .replace('&', 'and')
 .replace('.', '')
 .replace(' ', '')
)


# Satisfied, we create a method called `clean_county` that normalizes an input county.

# In[7]:


def clean_county(county):
    return (county
            .lower()
            .strip()
            .replace(' county', '')
            .replace(' parish', '')
            .replace('&', 'and')
            .replace(' ', '')
            .replace('.', ''))


# We may now verify that the `clean_county` method produces matching counties for all the counties in both tables:

# In[8]:


([clean_county(county) for county in state['County']],
 [clean_county(county) for county in population['County']]
)


# Because each county in both tables has the same transformed representation, we may successfully join the two tables using the transformed county names.

# ## String Methods in pandas
# 
# In the code above we used a loop to transform each county name. `pandas` Series objects provide a convenient way to apply string methods to each item in the series. First, the series of county names in the `state` table:

# In[9]:


state['County']


# The `.str` property on `pandas` Series exposes the same string methods as Python does. Calling a method on the `.str` property calls the method on each item in the series.

# In[10]:


state['County'].str.lower()


# This allows us to transform each string in the series without using a loop.

# In[11]:


(state['County']
 .str.lower()
 .str.strip()
 .str.replace(' parish', '')
 .str.replace(' county', '')
 .str.replace('&', 'and')
 .str.replace('.', '')
 .str.replace(' ', '')
)


# We save the transformed counties back into their originating tables:

# In[12]:


state['County'] = (state['County']
 .str.lower()
 .str.strip()
 .str.replace(' parish', '')
 .str.replace(' county', '')
 .str.replace('&', 'and')
 .str.replace('.', '')
 .str.replace(' ', '')
)

population['County'] = (population['County']
 .str.lower()
 .str.strip()
 .str.replace(' parish', '')
 .str.replace(' county', '')
 .str.replace('&', 'and')
 .str.replace('.', '')
 .str.replace(' ', '')
)


# Now, the two tables contain the same string representation of the counties:

# In[13]:


state


# In[14]:


population


# It is simple to join these tables once the counties match.

# In[15]:


state.merge(population, on='County')


# ## Summary
# 
# Python's string methods form a set of simple and useful operations for string manipulation. `pandas` Series implement the same methods that apply the underlying Python method to each string in the series.
# 
# You may find the complete docs on Python's `string` methods [here](https://docs.python.org/3/library/stdtypes.html#string-methods) and the docs on Pandas `str` methods [here](https://pandas.pydata.org/pandas-docs/stable/text.html#method-summary).
