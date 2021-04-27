#!/usr/bin/env python
# coding: utf-8

# # Data Tables in Python
# 
# Data tables, like the datasets we have worked with in Data 8, are one of the
# most common and useful forms of data for analysis. We introduce data 
# manipulation using `pandas`, the standard Python library for working with
# data tables. Although `pandas`'s syntax is more challenging to use than the
# `datascience` package used in Data 8, `pandas` provides significant performance
# improvements and is the current tool of choice in both industry and academia.
# 
# It is more important that you understand the types of useful operations on data
# than the exact details of `pandas` syntax. For example, knowing when to perform a
# group-by is generally more useful than knowing how to call the `pandas` function
# to group data. Since this chapter contains many snippets of code, we
# encourage you to read through this chapter twice: once to understand the syntax
# and once to understand when each operation is appropriate.
# 
# Because we will cover only the most commonly used `pandas` functions in this
# textbook, you should bookmark the [`pandas` documentation][docs] for reference
# when you conduct your own data analyses.
# 
# We begin by talking about the types of dataset structures that `pandas` can read.
# Then, we introduce indexes, grouping, apply, and strings.
# 
# [docs]: http://pandas.pydata.org/pandas-docs/stable/
# 

# 
# ```{toctree}
# :hidden:
# :titlesonly:
# 
# 
# pandas_indexes
# pandas_grouping_pivoting
# pandas_apply_strings_plotting
# pandas_joins
# ```
# 
