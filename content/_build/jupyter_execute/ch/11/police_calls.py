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


# # Cleaning the Calls Dataset
# 
# We will use the Berkeley Police Department's publicly available datasets to demonstrate data cleaning techniques. We have downloaded the [Calls for Service dataset][calls] and [Stops dataset][stops].
# 
# We can use the `ls` shell command with the `-lh` flags to see more details about the files:
# 
# [calls]: https://data.cityofberkeley.info/Public-Safety/Berkeley-PD-Calls-for-Service/k2nh-s5h5
# [stops]: https://data.cityofberkeley.info/Public-Safety/Berkeley-PD-Stop-Data/6e9j-pj9p

# In[2]:


get_ipython().system('ls -lh data/')


# The command above shows the data files and their file sizes. This is especially useful because we now know the files are small enough to load into memory. As a rule of thumb, it is usually safe to load a file into memory that is around one fourth of the total memory capacity of the computer. For example, if a computer has 4GB of RAM we should be able to load a 1GB CSV file in `pandas`. To handle larger datasets we will need additional computational tools that we will cover later in this book.
# 
# Notice the use of the exclamation point before `ls`. This tells Jupyter that the next line of code is a shell command, not a Python expression. We can run any available shell command in Jupyter using `!`:

# In[3]:


# The `wc` shell command shows us how many lines each file has.
# We can see that the `stops.json` file has the most lines (29852).
get_ipython().system('wc -l data/*')


# ## Understanding the Data Generation
# 
# We will state important questions you should ask of all datasets before data cleaning or processing. These questions are related to how the data were generated, so data cleaning will usually **not** be able to resolve issues that arise here.
# 
# **What do the data contain?** The website for the Calls for Service data states that the dataset describes "crime incidents (not criminal reports) within the last 180 days". Further reading reveals that "not all calls for police service are included (e.g. Animal Bite)".
# 
# The website for the Stops data states that the dataset contains data on all "vehicle detentions (including bicycles) and pedestrian detentions (up to five persons)" since January 26, 2015.
# 
# **Are the data a census?** This depends on our population of interest. For example, if we are interested in calls for service within the last 180 days for crime incidents then the Calls dataset is a census. However, if we are interested in calls for service within the last 10 years the dataset is clearly not a census. We can make similar statements about the Stops dataset since the data collection started on January 26, 2015.
# 
# **If the data form a sample, is it a probability sample?** If we are investigating a period of time that the data do not have entries for, the data do not form a probability sample since there is no randomness involved in the data collection process — we have all data for certain time periods but no data for others.
# 
# **What limitations will this data have on our conclusions?** Although we will ask this question at each step of our data processing, we can already see that our data impose important limitations. The most important limitation is that we cannot make unbiased estimations for time periods not covered by our datasets.

# Let's now clean the Calls dataset. The `head` shell command prints the first five lines of the file.

# In[4]:


get_ipython().system('head data/Berkeley_PD_-_Calls_for_Service.csv')


# It appears to be a comma-separated values (CSV) file, though it's hard to tell whether the entire file is formatted properly. We can use `pd.read_csv` to read in the file as a DataFrame. If `pd.read_csv` errors, we will have to dig deeper and manually resolve formatting issues. Fortunately, `pd.read_csv` successfully returns a DataFrame:

# In[5]:


calls = pd.read_csv('data/Berkeley_PD_-_Calls_for_Service.csv')
calls


# Based on the output above, the resulting DataFrame looks reasonably well-formed since the columns are properly named and the data in each column seems to be entered consistently. What data does each column contain? We can look at the dataset website:
# 
# | Column         | Description                            | Type        |
# | ------         | -----------                            | ----        |
# | CASENO         | Case Number                            | Number      |
# | OFFENSE        | Offense Type                           | Plain Text  |
# | EVENTDT        | Date Event Occurred                    | Date & Time |
# | EVENTTM        | Time Event Occurred                    | Plain Text  |
# | CVLEGEND       | Description of Event                   | Plain Text  |
# | CVDOW          | Day of Week Event Occurred             | Number      |
# | InDbDate       | Date dataset was updated in the portal | Date & Time |
# | Block_Location | Block level address of event           | Location    |
# | BLKADDR        |                                        | Plain Text  |
# | City           |                                        | Plain Text  |
# | State          |                                        | Plain Text  |

# On the surface the data looks easy to work with. However, before starting data analysis we must answer the following questions:
# 
# 1. **Are there missing values in the dataset?** This question is important because missing values can represent many different things. For example, missing addresses could mean that locations were removed to protect anonymity, or that some respondents chose not to answer a survey question, or that a recording device broke.
# 1. **Are there any missing values that were filled in (e.g. a 999 for unknown age or 12:00am for unknown date)?** These will clearly impact analysis if we ignore them.
# 1. **Which parts of the data were entered by a human?** As we will soon see, human-entered data is filled with inconsistencies and mispellings.
# 
# Although there are plenty more checks to go through, these three will suffice for many cases. See the [Quartz bad data guide](https://github.com/Quartz/bad-data-guide) for a more complete list of checks.

# ### Are there missing values?
# 
# This is a simple check in `pandas`:

# In[6]:


# True if row contains at least one null value
null_rows = calls.isnull().any(axis=1)
calls[null_rows]


# It looks like 27 calls didn't have a recorded address in BLKADDR. Unfortunately, the data description isn't very clear on how the locations were recorded. We know that all of these calls were made for events in Berkeley, so we can likely assume that the addresses for these calls were originally somewhere in Berkeley.

# ### Are there any missing values that were filled in?
# 
# From the missing value check above we can see that the Block_Location column has Berkeley, CA recorded if the location was missing.
# 
# In addition, an inspection of the `calls` table shows us that the EVENTDT column has the correct dates but records 12am for all of its times. Instead, the times are in the EVENTTM column.

# In[7]:


# Show the first 7 rows of the table again for reference
calls.head(7)


# As a data cleaning step, we want to merge the EVENTDT and EVENTTM columns to record both date and time in one field. If we define a function that takes in a DF and returns a new DF, we can later use [`pd.pipe`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.pipe.html) to apply all transformations in one go.

# In[8]:


def combine_event_datetimes(calls):
    combined = pd.to_datetime(
        # Combine date and time strings
        calls['EVENTDT'].str[:10] + ' ' + calls['EVENTTM'],
        infer_datetime_format=True,
    )
    return calls.assign(EVENTDTTM=combined)

# To peek at the result without mutating the calls DF:
calls.pipe(combine_event_datetimes).head(2)


# ### Which parts of the data were entered by a human?
# 
# It looks like most of the data columns are machine-recorded, including the date, time, day of week, and location of the event.
# 
# In addition, the OFFENSE and CVLEGEND columns appear to contain consistent values. We can check the unique values in each column to see if anything was misspelled:

# In[9]:


calls['OFFENSE'].unique()


# In[10]:


calls['CVLEGEND'].unique()


# Since each value in these columns appears to be spelled correctly, we won't have to perform any corrections on these columns.
# 
# We also check the BLKADDR column for inconsistencies and find that sometimes an address is recorded (e.g. 2500 LE CONTE AVE) but other times a cross street is recorded (e.g. ALLSTON WAY & FIFTH ST). This suggests that a human entered this data in and this column will be difficult to use for analysis. Fortunately we can use the latitude and longitude of the event instead of the street address.

# In[11]:


calls['BLKADDR'][[0, 5001]]


# ### Final Touchups
# 
# This dataset seems almost ready for analysis. The Block_Location column seems to contain strings that record address, latitude, and longitude. We will want to separate the latitude and longitude for easier use.

# In[12]:


def split_lat_lon(calls):
    return calls.join(
        calls['Block_Location']
        # Get coords from string
        .str.split('\n').str[2]
        # Remove parens from coords
        .str[1:-1]
        # Split latitude and longitude
        .str.split(', ', expand=True)
        .rename(columns={0: 'Latitude', 1: 'Longitude'})
    )

calls.pipe(split_lat_lon).head(2)


# Then, we can match the day of week number with its weekday:

# In[13]:


# This DF contains the day for each number in CVDOW
day_of_week = pd.read_csv('data/cvdow.csv')
day_of_week


# In[14]:


def match_weekday(calls):
    return calls.merge(day_of_week, on='CVDOW')
calls.pipe(match_weekday).head(2)


# We'll drop columns we no longer need:

# In[15]:


def drop_unneeded_cols(calls):
    return calls.drop(columns=['CVDOW', 'InDbDate', 'Block_Location', 'City',
                               'State', 'EVENTDT', 'EVENTTM'])


# Finally, we'll pipe the `calls` DF through all the functions we've defined:

# In[16]:


calls_final = (calls.pipe(combine_event_datetimes)
               .pipe(split_lat_lon)
               .pipe(match_weekday)
               .pipe(drop_unneeded_cols))
calls_final


# The Calls dataset is now ready for further data analysis. In the next section, we will clean the Stops dataset.

# In[17]:


# Save data to CSV for other chapters
calls_final.to_csv('data/calls.csv', index=False)

