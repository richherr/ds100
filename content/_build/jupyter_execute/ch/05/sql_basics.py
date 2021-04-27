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


# In[2]:


# Creating a table
sql_expr = """
CREATE TABLE prices(
    retailer TEXT,
    product TEXT,
    price FLOAT);
"""
result = sqlite_engine.execute(sql_expr)


# In[3]:


# Inserting records into the table
sql_expr = """
INSERT INTO prices VALUES 
  ('Best Buy', 'Galaxy S9', 719.00),
  ('Best Buy', 'iPod', 200.00),
  ('Amazon', 'iPad', 450.00),
  ('Amazon', 'Battery pack',  24.87),
  ('Amazon', 'Chromebook', 249.99),
  ('Target', 'iPod', 215.00),
  ('Target', 'Surface Pro', 799.00),
  ('Target', 'Google Pixel 2', 659.00),
  ('Walmart', 'Chromebook', 238.79);
"""
result = sqlite_engine.execute(sql_expr)


# In[4]:


import pandas as pd

prices = pd.DataFrame([['Best Buy', 'Galaxy S9', 719.00],
                   ['Best Buy', 'iPod', 200.00],
                   ['Amazon', 'iPad', 450.00],
                   ['Amazon', 'Battery pack', 24.87],
                   ['Amazon', 'Chromebook', 249.99],
                   ['Target', 'iPod', 215.00],
                   ['Target', 'Surface Pro', 799.00],
                   ['Target', 'Google Pixel 2', 659.00],
                   ['Walmart', 'Chromebook', 238.79]],
                 columns=['retailer', 'product', 'price'])


# # SQL
# 
# **SQL** (Structured Query Language) is a programming language that has operations to define, logically organize, manipulate, and perform calculations on data stored in a relational database management system (RDBMS).
# 
# SQL is a declarative language. This means that the user only needs to specify *what* kind of data they want, not *how* to obtain it. An example is shown below, with an imperative example for comparison:
# 
# - **Declarative**: Compute the table with columns "x" and "y" from table "A" where the values in "y" are greater than 100.00.
# - **Imperative**: For each record in table "A", check if the record contains a value of "y" greater than 100. If so, then store the record's "x" and "y" attributes in a new table. Return the new table.
# 
# In this chapter, we will write SQL queries as Python strings, then use `pandas` to execute the SQL query and read the result into a `pandas` DataFrame. As we walk through the basics of SQL syntax, we'll also occasionally show `pandas` equivalents for comparison purposes.

# ## Executing SQL Queries through `pandas`
# 
# To execute SQL queries from Python, we will connect to a database using the [sqlalchemy](http://docs.sqlalchemy.org/en/latest/core/tutorial.html) library. Then we can use the `pandas` function [pd.read_sql](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_sql.html) to execute SQL queries through this connection.

# In[5]:


import sqlalchemy

# pd.read_sql takes in a parameter for a SQLite engine, which we create below
sqlite_uri = "sqlite:///sql_basics.db"
sqlite_engine = sqlalchemy.create_engine(sqlite_uri)


# This database contains one relation: `prices`. To display the relation we run a SQL query. Calling `read_sql` will execute the SQL query on the RDBMS, then return the results in a `pandas` DataFrame.

# In[6]:


sql_expr = """
SELECT * 
FROM prices
"""
pd.read_sql(sql_expr, sqlite_engine)


# Later in this section we will compare SQL queries with `pandas` method calls so we've created an identical DataFrame in `pandas`.

# In[7]:


prices


# ## SQL Syntax
# 
# All SQL queries take the general form below:
# ```SQL
# SELECT [DISTINCT] <column expression list>
# FROM <relation>
# [WHERE <predicate>]
# [GROUP BY <column list>]
# [HAVING <predicate>]
# [ORDER BY <column list>]
# [LIMIT <number>]
# ```
# 
# Note that:
# 
# 1. **Everything in \[square brackets\] is optional.** A valid SQL query only needs a `SELECT` and a `FROM` statement.
# 2. **SQL SYNTAX IS GENERALLY WRITTEN IN CAPITAL LETTERS.** Although capitalization isn't required, it is common practice to write SQL syntax in capital letters. It also helps to visually structure your query for others to read.
# 3. `FROM` query blocks can reference one or more tables, although in this section we will only look at one table at a time for simplicity.

# ### SELECT and FROM
# 
# The two mandatory statements in a SQL query are:
# 
# * `SELECT` indicates the columns that we want to view.
# * `FROM` indicates the tables from which we are selecting these columns.
# 
# To display the entire `prices` table, we run:

# In[8]:


sql_expr = """
SELECT * 
FROM prices
"""
pd.read_sql(sql_expr, sqlite_engine)


# `SELECT *` returns every column in the original relation. To display only the retailers that are represented in `prices`, we add the `retailer` column to the `SELECT` statement.

# In[9]:


sql_expr = """
SELECT retailer
FROM prices
"""
pd.read_sql(sql_expr, sqlite_engine)


# If we want a list of unique retailers, we can call the `DISTINCT` function to omit repeated values.

# In[10]:


sql_expr = """
SELECT DISTINCT(retailer)
FROM prices
"""
pd.read_sql(sql_expr, sqlite_engine)


# This would be the functional equivalent of the following `pandas` code:

# In[11]:


prices['retailer'].unique()


# Each RDBMS comes with its own set of functions that can be applied to attributes in the `SELECT` list, such as comparison operators, mathematical functions and operators, and string functions and operators. In Data 100 we use PostgreSQL, a mature RDBMS that comes with hundreds of such functions. The complete list is available [here](https://www.postgresql.org/docs/9.2/static/functions.html). Keep in mind that each RDBMS has a different set of functions for use in `SELECT`.
# 
# The following code converts all retailer names to uppercase and halves the product prices.

# In[12]:


sql_expr = """
SELECT
    UPPER(retailer) AS retailer_caps,
    product,
    price / 2 AS half_price
FROM prices
"""
pd.read_sql(sql_expr, sqlite_engine)


# Notice that we can **alias** the columns (assign another name) with `AS` so that the columns appear with this new name in the output table. This does not modify the names of the columns in the source relation.

# ### WHERE
# 
# The `WHERE` clause allows us to specify certain constraints for the returned data; these constraints are often referred to as **predicates**. For example, to retrieve only gadgets that are under $500:

# In[13]:


sql_expr = """
SELECT *
FROM prices
WHERE price < 500
"""
pd.read_sql(sql_expr, sqlite_engine)


# We can also use the operators `AND`, `OR`, and `NOT` to further constrain our SQL query. To find an item on Amazon without a battery pack under $300, we write:

# In[14]:


sql_expr = """
SELECT *
FROM prices
WHERE retailer = 'Amazon'
    AND NOT product = 'Battery pack'
    AND price < 300
"""
pd.read_sql(sql_expr, sqlite_engine)


# The equivalent operation in `pandas` is:

# In[15]:


prices[(prices['retailer'] == 'Amazon') 
   & ~(prices['product'] == 'Battery pack')
   & (prices['price'] <= 300)]


# There's a subtle difference that's worth noting: the index of the Chromebook in the SQL query is 0, whereas the corresponding index in the DataFrame is 4. This is because SQL queries always return a new table with indices counting up from 0, whereas `pandas` subsets a portion of the DataFrame `prices` and returns it with the original indices. We can use [pd.DataFrame.reset_index](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.reset_index.html) to reset the indices in `pandas`.

# ### Aggregate Functions
# 
# So far, we've only worked with data from the existing rows in the table; that is, all of our returned tables have been some subset of the entries found in the table. But to conduct data analysis, we'll want to compute aggregate values over our data. In SQL, these are called **aggregate functions**. 
# 
# If we want to find the average price of all gadgets in the `prices` relation:

# In[16]:


sql_expr = """
SELECT AVG(price) AS avg_price
FROM prices
"""
pd.read_sql(sql_expr, sqlite_engine)


# Equivalently, in `pandas`:
# 

# In[17]:


prices['price'].mean()


# A complete list of PostgreSQL aggregate functions can be found [here](https://www.postgresql.org/docs/9.2/static/functions.html). Though we're using PostgreSQL as our primary version of SQL in Data 100, keep in mind that there are many other variations of SQL (MySQL, SQLite, etc.) that use different function names and have different functions available.

# ### GROUP BY and HAVING
# 
# With aggregate functions, we can execute more complicated SQL queries. To operate on more granular aggregate data, we can use the following two clauses:
# - `GROUP BY` takes a list of columns and groups the table like the [pd.DataFrame.groupby](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.groupby.html) function in `pandas`.
# - `HAVING` is functionally similar to `WHERE`, but is used exclusively to apply predicates to aggregated data. (Note that in order to use `HAVING`, it must be preceded by a `GROUP BY` clause.)
# 
# **Important**: When using `GROUP BY`, all columns in the `SELECT` clause must be either listed in the `GROUP BY` clause or have an aggregate function applied to them.
# 
# We can use these statements to find the maximum price at each retailer.

# In[18]:


sql_expr = """
SELECT retailer, MAX(price) as max_price
FROM prices
GROUP BY retailer
"""
pd.read_sql(sql_expr, sqlite_engine)


# Let's say we have a client with expensive taste and only want to find retailers that sell gadgets over $700. Note that we must use `HAVING` to define predicates on aggregated columns; we can't use `WHERE` to filter an aggregated column. To compute a list of retailers and accompanying prices that satisfy our needs, we run:

# In[19]:


sql_expr = """
SELECT retailer, MAX(price) as max_price
FROM prices
GROUP BY retailer
HAVING max_price > 700
"""
pd.read_sql(sql_expr, sqlite_engine)


# For comparison, we recreate the same table in `pandas`:

# In[20]:


max_prices = prices.groupby('retailer').max()
max_prices.loc[max_prices['price'] > 700, ['price']]


# ### ORDER BY and LIMIT
# 
# These clauses allow us to control the presentation of the data:
# - `ORDER BY` lets us present the data in lexicographic order of column values. By default, ORDER BY uses ascending order (`ASC`) but we can specify descending order using `DESC`.
# - `LIMIT` controls how many tuples are displayed.
# 
# Let's display the three cheapest items in our `prices` table:

# In[21]:


sql_expr = """
SELECT *
FROM prices
ORDER BY price ASC
LIMIT 3
"""
pd.read_sql(sql_expr, sqlite_engine)


# Note that we didn't have to include the `ASC` keyword since `ORDER BY` returns data in ascending order by default.
# For comparison, in `pandas`:

# In[22]:


prices.sort_values('price').head(3)


# (Again, we see that the indices are out of order in the `pandas` DataFrame. As before, `pandas` returns a view on our DataFrame `prices`, whereas SQL is displaying a new table each time that we execute a query.)

# ### Conceptual SQL Evaluation
# 
# Clauses in a SQL query are executed in a specific order. Unfortunately, this order differs from the order that the clauses are written in a SQL query. From first executed to last:
# 
# 1. `FROM`: One or more source tables
# 2. `WHERE`: Apply selection qualifications (eliminate rows)
# 3. `GROUP BY`: Form groups and aggregate
# 4. `HAVING`: Eliminate groups
# 5. `SELECT`: Select columns
# 
# **Note on `WHERE` vs. `HAVING`**: Since the `WHERE` clause is processed before applying `GROUP BY`, the `WHERE` clause cannot make use of aggregated values. To define predicates based on aggregated values, we must use the `HAVING` clause.

# ## Summary
# 
# We have introduced SQL syntax and the most important SQL statements needed to conduct data analysis using a relational database management system.
