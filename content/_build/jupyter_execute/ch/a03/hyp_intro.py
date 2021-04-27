#!/usr/bin/env python
# coding: utf-8

# # Statistical Inference Review
# 
# This appendix reviews inference using permutation tests and the bootstrap method. This content is also covered in chapters [12][12] and [13][13] of the Data 8 textbook.
# 
# [12]: https://www.inferentialthinking.com/chapters/12/Comparing_Two_Samples.html
# [13]: https://www.inferentialthinking.com/chapters/13/Estimation.html
# 
# Although data scientists often work with individual samples of data, we are
# almost always interested in making generalizations about the population that
# the data were collected from. This chapter discusses methods for _statistical
# inference_, the process of drawing conclusions about a entire population using
# a dataset.
# 
# Statistical inference primarily leans on two methods: hypothesis tests and
# confidence intervals. In the recent past these methods relied heavily on normal
# theory, a branch of statistics that requires substantial assumptions about the
# population. Today, the rapid rise of powerful computing resources has
# enabled a new class of methods based on _resampling_ that generalize to many
# types of populations.

# 
# ```{toctree}
# :hidden:
# :titlesonly:
# 
# 
# hyp_introduction
# hyp_introduction_part2
# ```
# 
