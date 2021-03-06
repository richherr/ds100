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


# # Regex and Python
# 
# In this section, we introduce regex usage in Python using the built-in `re` module. Since we only cover a few of the most commonly used methods, you will find it useful to consult [the official documentation on the `re` module](https://docs.python.org/3/library/re.html) as well.

# ## `re.search`
# 
# `re.search(pattern, string)` searches for a match of the regex `pattern` anywhere in `string`. It returns a truthy match object if the pattern is found; it returns `None` if not.

# In[3]:


phone_re = r"[0-9]{3}-[0-9]{3}-[0-9]{4}"
text  = "Call me at 382-384-3840."
match = re.search(phone_re, text)
match


# Although the returned match object has a variety of useful properties, we most commonly use `re.search` to test whether a pattern appears in a string.

# In[4]:


if re.search(phone_re, text):
    print("Found a match!")


# In[5]:


if re.search(phone_re, 'Hello world'):
    print("No match; this won't print")


# Another commonly used method, `re.match(pattern, string)`, behaves the same as `re.search` but only checks for a match at the start of `string` instead of a match anywhere in the string.

# ## `re.findall`
# 
# We use `re.findall(pattern, string)` to extract substrings that match a regex. This method returns a list of all matches of `pattern` in `string`.

# In[10]:


gmail_re = r'[a-zA-Z0-9]+@gmail\.com'
text = '''
From: email1@gmail.com
To: email2@yahoo.com and email3@gmail.com
'''
re.findall(gmail_re, text)


# ## Regex Groups
# 
# 

# Using **regex groups**, we specify subpatterns to extract from a regex by wrapping the subpattern in parentheses `( )`. When a regex contains regex groups, `re.findall` returns a list of tuples that contain the subpattern contents.
# 
# For example, the following familiar regex extracts phone numbers from a string:

# In[11]:


phone_re = r"[0-9]{3}-[0-9]{3}-[0-9]{4}"
text  = "Sam's number is 382-384-3840 and Mary's is 123-456-7890."
re.findall(phone_re, text)


# To split apart the individual three or four digit components of a phone number, we can wrap each digit group in parentheses.

# In[12]:


# Same regex with parentheses around the digit groups
phone_re = r"([0-9]{3})-([0-9]{3})-([0-9]{4})"
text  = "Sam's number is 382-384-3840 and Mary's is 123-456-7890."
re.findall(phone_re, text)


# As promised, `re.findall` returns a list of tuples containing the individual components of the matched phone numbers.

# ## `re.sub`
# 
# `re.sub(pattern, replacement, string)` replaces all occurrences of `pattern` with `replacement` in the provided `string`. This method behaves like the Python string method `str.sub` but uses a regex to match patterns.
# 
# In the code below, we alter the dates to have a common format by substituting the date separators with a dash.

# In[14]:


messy_dates = '03/12/2018, 03.13.18, 03/14/2018, 03:15:2018'
regex = r'[/.:]'
re.sub(regex, '-', messy_dates)


# ## `re.split`
# 
# `re.split(pattern, string)` splits the input `string` each time the regex `pattern` appears. This method behaves like the Python string method `str.split` but uses a regex to make the split.
# 
# In the code below, we use `re.split` to split chapter names from their page numbers in a table of contents for a book.

# In[21]:


toc = '''
PLAYING PILGRIMS============3
A MERRY CHRISTMAS===========13
THE LAURENCE BOY============31
BURDENS=====================55
BEING NEIGHBORLY============76
'''.strip()

# First, split into individual lines
lines = re.split('\n', toc)
lines


# In[22]:


# Then, split into chapter title and page number
split_re = r'=+' # Matches any sequence of = characters
[re.split(split_re, line) for line in lines]


# ## Regex and pandas
# 
# Recall that `pandas` Series objects have a `.str` property that supports string manipulation using Python string methods. Conveniently, the `.str` property also supports some functions from the `re` module. We demonstrate basic regex usage in `pandas`, leaving the complete method list to [the `pandas` documentation on string methods](https://pandas.pydata.org/pandas-docs/stable/text.html).
# 
# We've stored the text of the first five sentences of the novel *Little Women* in the DataFrame below. We can use the string methods that `pandas` provides to extract the spoken dialog in each sentence.

# In[25]:


text = '''
"Christmas won't be Christmas without any presents," grumbled Jo, lying on the rug.
"It's so dreadful to be poor!" sighed Meg, looking down at her old dress.
"I don't think it's fair for some girls to have plenty of pretty things, and other girls nothing at all," added little Amy, with an injured sniff.
"We've got Father and Mother, and each other," said Beth contentedly from her corner.
The four young faces on which the firelight shone brightened at the cheerful words, but darkened again as Jo said sadly, "We haven't got Father, and shall not have him for a long time."
'''.strip()
little = pd.DataFrame({
    'sentences': text.split('\n')
})


# In[26]:


little


# Since spoken dialog lies within double quotation marks, we create a regex that captures a double quotation mark, a sequence of any characters except a double quotation mark, and the closing quotation mark.

# In[32]:


quote_re = r'"[^"]+"'
little['sentences'].str.findall(quote_re)


# Since the `Series.str.findall` method returns a list of matches, `pandas` also provides `Series.str.extract` and `Series.str.extractall` method to extract matches into a Series or DataFrame. These methods require the regex to contain at least one regex group.

# In[36]:


# Extract text within double quotes
quote_re = r'"([^"]+)"'
spoken = little['sentences'].str.extract(quote_re)
spoken


# We can add this series as a column of the `little` DataFrame:

# In[38]:


little['dialog'] = spoken
little


# We can confirm that our string manipulation behaves as expected for the last sentence in our DataFrame by printing the original and extracted text:

# In[40]:


print(little.loc[4, 'sentences'])


# In[41]:


print(little.loc[4, 'dialog'])


# ## Summary
# 
# The `re` module in Python provides a useful group of methods for manipulating text using regular expressions. When working with DataFrames, we often use the analogous string manipulation methods implemented in `pandas`.
# 
# For the complete documentation on the `re` module, see https://docs.python.org/3/library/re.html
# 
# For the complete documentation on `pandas` string methods, see https://pandas.pydata.org/pandas-docs/stable/text.html
# 
