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


# # The Data Science Lifecycle
# 
# In data science, we use large and diverse data sets to make conclusions about
# the world. In this book we discuss principles and techniques of
# data science through the two lens of computational and inferential thinking.
# Practically speaking, this involves the following process:
# 
# 1. Formulating a question or problem
# 2. Acquiring and cleaning data
# 3. Conducting exploratory data analysis
# 4. Using prediction and inference to draw conclusions
# 
# It is quite common for more questions and problems to emerge after the last
# step of this process, so we repeatedly engage in this procedure to
# discover new characteristics of our data. This positive feedback loop is so
# central to our work that we call it the **data science lifecycle**.
# 
# While simple to state, the data science lifecycle takes training and practice
# to do well. In fact, each topic in this book revolves around a piece of this
# lifecycle. We think learning to do data science is both challenging and
# rewarding – we'll show you by starting with an example.

# 
# ```{toctree}
# :hidden:
# :titlesonly:
# 
# 
# lifecycle_students_1
# lifecycle_students_2
# lifecycle_students_3
# ```
# 
