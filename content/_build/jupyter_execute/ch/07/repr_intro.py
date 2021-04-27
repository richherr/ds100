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


# In[ ]:


def df_interact(df, nrows=7, ncols=7):
    '''
    Outputs sliders that show rows and columns of df
    '''
    def peek(row=0, col=0):
        return df.iloc[row:row + nrows, col:col + ncols]

    row_arg = (0, len(df), nrows) if len(df) > nrows else fixed(0)
    col_arg = ((0, len(df.columns), ncols)
               if len(df.columns) > ncols else fixed(0))
    
    interact(peek, row=row_arg, col=col_arg)
    print('({} rows, {} columns) total'.format(df.shape[0], df.shape[1]))

def display_df(df, rows=pd.options.display.max_rows,
               cols=pd.options.display.max_columns):
    with pd.option_context('display.max_rows', rows,
                           'display.max_columns', cols):
        display(df)


# # Data Representation

# On one level, data are just bytes in a computer disk. But, these bytes can be very useful when we attach them to things that happen in the real world. Drivers in Los Angeles use a service called Sigalert [^sigalert] to see which freeways have jams (although in Los Angeles, the answer is often "all of them"). Sensors installed on freeway roads record the speed of cars passing by. Sigalert takes that data, makes a map, and displays it to users:
# 
# ```{image} sigalert.png
# :alt: sigalert.png
# :align: center
# ```
# 
# Sigalert displays data that **represent** traffic conditions. Data scientists navigate a few levels of data representation. The data's **structure** and **format** refers to the way the computer represents the data. SQL databases store data in a table structure, but other kinds of data structures exist. **Data semantics** refer to the real-world meaning that we as data scientists assign to each datum. Every data analysis begins by understanding what the data represent.

# [^sigalert]: [https://www.sigalert.com/](https://www.sigalert.com/)

# 
# ```{toctree}
# :hidden:
# :titlesonly:
# 
# 
# repr_structure
# repr_semantics
# repr_data_types
# repr_granularity
# ```
# 
