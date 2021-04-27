#!/usr/bin/env python
# coding: utf-8

# In[5]:


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


# # Granularity
# 
# The granularity of your data is what each record in your data represents. For example, in the Calls dataset each record represents a single case of a police call.

# In[6]:


calls = pd.read_csv('data/calls.csv')
calls.head()


# In the Stops dataset, each record represents a single incident of a police stop.

# In[7]:


stops = pd.read_csv('data/stops.csv', parse_dates=[1], infer_datetime_format=True)
stops.head()


# On the other hand, we could have received the Stops data in the following format:

# In[8]:


(stops
 .groupby(stops['Call Date/Time'].dt.date)
 .size()
 .rename('Num Incidents')
 .to_frame()
)


# In this case, each record in the table corresponds to a single date instead of a single incident. We would describe this table as having a coarser granularity than the one above. It's important to know the granularity of your data because it determines what kind of analyses you can perform. Generally speaking, too fine of a granularity is better than too coarse; while we can use grouping and pivoting to change a fine granularity to a coarse one, we have few tools to go from coarse to fine.

# ## Granularity Checklist
# 
# You should have answers to the following questions after looking at the granularity of your datasets. We will answer them for the Calls and Stops datasets.
# 
# **What does a record represent?**
# 
# In the Calls dataset, each record represents a single case of a police call. In the Stops dataset, each record represents a single incident of a police stop.
# 
# **Do all records capture granularity at the same level? (Sometimes a table will contain summary rows.)**
# 
# Yes, for both Calls and Stops datasets.
# 
# **If the data were aggregated, how was the aggregation performed? Sampling and averaging are are common aggregations.**
# 
# No aggregations were performed as far as we can tell for the datasets. We do keep in mind that in both datasets, the location is entered as a block location instead of a specific address.
# 
# **What kinds of aggregations can we perform on the data?**
# 
# For example, it's often useful to aggregate individual people to demographic groups or individual events to totals across time.
# 
# In this case, we can aggregate across various granularities of date or time. For example, we can find the most common hour of day for incidents with aggregation. We might also be able to aggregate across event locations to find the regions of Berkeley with the most incidents.
