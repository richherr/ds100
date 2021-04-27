#!/usr/bin/env python
# coding: utf-8

# # Regular Expressions
# 
# In this section we introduce regular expressions, an important tool to specify patterns in strings.

# ## Motivation
# 
# In a larger piece of text, many useful substrings come in a specific format. For instance, the sentence below contains a U.S. phone number.
# 
# `"give me a call, my number is 123-456-7890."`
# 
# The phone number contains the following pattern:
# 
# 1. Three numbers
# 1. Followed by a dash
# 1. Followed by three numbers
# 1. Followed by a dash
# 1. Followed by four Numbers
# 
# Given a free-form segment of text, we might naturally wish to detect and extract the phone numbers. We may also wish to extract specific pieces of the phone numbers—for example, by extracting the area code we may deduce the locations of individuals mentioned in the text.
# 
# To detect whether a string contains a phone number, we may attempt to write a method like the following:

# In[121]:


def is_phone_number(string):
    
    digits = '0123456789'
    
    def is_not_digit(token):
        return token not in digits 
    
    # Three numbers
    for i in range(3):
        if is_not_digit(string[i]):
            return False
    
    # Followed by a dash
    if string[3] != '-':
        return False
    
    # Followed by three numbers
    for i in range(4, 7):
        if is_not_digit(string[i]):
            return False
        
    # Followed by a dash    
    if string[7] != '-':
        return False
    
    # Followed by four numbers
    for i in range(8, 12):
        if is_not_digit(string[i]):
            return False
    
    return True


# In[122]:


is_phone_number("382-384-3840")


# In[123]:


is_phone_number("phone number")


# The code above is unpleasant and verbose. Rather than manually loop through the characters of the string, we would prefer to specify a pattern and command Python to match the pattern.
# 
# **Regular expressions** (often abbreviated **regex**) conveniently solve this exact problem by allowing us to create general patterns for strings. Using a regular expression, we may re-implement the `is_phone_number` method in two short lines of Python:

# In[124]:


import re

def is_phone_number(string):
    regex = r"[0-9]{3}-[0-9]{3}-[0-9]{4}"
    return re.search(regex, string) is not None

is_phone_number("382-384-3840")


# In the code above, we use the regex `[0-9]{3}-[0-9]{3}-[0-9]{4}` to match phone numbers. Although cryptic at a first glance, the syntax of regular expressions is fortunately much simpler to learn than the Python language itself; we introduce nearly all of the syntax in this section alone.
# 
# We will also introduce the built-in Python module `re` that performs string operations using regexes. 

# ## Regex Syntax

# We start with the syntax of regular expressions. In Python, regular expressions are most commonly stored as raw strings. Raw strings behave like normal Python strings without special handling for backslashes.
# 
# For example, to store the string `hello \ world` in a normal Python string, we must write:

# In[125]:


# Backslashes need to be escaped in normal Python strings
some_string = 'hello \\ world'
print(some_string)


# Using a raw string removes the need to escape the backslash:

# In[126]:


# Note the `r` prefix on the string
some_raw_string = r'hello \ world'
print(some_raw_string)


# Since backslashes appear often in regular expressions, we will use raw strings for all regexes in this section.

# ## Literals
# 
# A **literal** character in a regular expression matches the character itself. For example, the regex `r"a"` will match any `"a"` in `"Say! I like green eggs and ham!"`. All alphanumeric characters and most punctuation characters are regex literals.

# In[127]:


def show_regex_match(text, regex):
    """
    Prints the string with the regex match highlighted.
    """
    print(re.sub(f'({regex})', r'\033[1;30;43m\1\033[m', text))


# In[128]:


# The show_regex_match method highlights all regex matches in the input string
regex = r"green"
show_regex_match("Say! I like green eggs and ham!", regex)


# In[129]:


show_regex_match("Say! I like green eggs and ham!", r"a")


# In the example above we observe that regular expressions can match patterns that appear anywhere in the input string. In Python, this behavior differs depending on the method used to match the regex—some methods only return a match if the regex appears at the start of the string; some methods return a match anywhere in the string.
# 
# Notice also that the `show_regex_match` method highlights all occurrences of the regex in the input string. Again, this differs depending on the Python method used—some methods return all matches while some only return the first match.
# 
# Regular expressions are case-sensitive. In the example below, the regex only matches the lowercase `s` in `eggs`, not the uppercase `S` in `Say`.

# In[130]:


show_regex_match("Say! I like green eggs and ham!", r"s")


# ## Wildcard Character
# 
# Some characters have special meaning in a regular expression. These meta characters allow regexes to match a variety of patterns.
# 
# In a regular expression, the period character `.` matches any character except a newline.

# In[131]:


show_regex_match("Call me at 382-384-3840.", r".all")


# To match only the literal period character we must escape it with a backslash:

# In[132]:


show_regex_match("Call me at 382-384-3840.", r"\.")


# By using the period character to mark the parts of a pattern that vary, we construct a regex to match phone numbers. For example, we may take our original phone number `382-384-3840` and replace the numbers with `.`, leaving the dashes as literals. This results in the regex `...-...-....`.

# In[133]:


show_regex_match("Call me at 382-384-3840.", "...-...-....")


# Since the period character matches all characters, however, the following input string will produce a spurious match.

# In[134]:


show_regex_match("My truck is not-all-blue.", "...-...-....")


# ## Character Classes
# 
# A **character class** matches a specified set of characters, allowing us to create more restrictive matches than the `.` character alone. To create a character class, wrap the set of desired characters in brackets `[ ]`.

# In[135]:


show_regex_match("I like your gray shirt.", "gr[ae]y")


# In[136]:


show_regex_match("I like your grey shirt.", "gr[ae]y")


# In[137]:


# Does not match; a character class only matches one character from a set
show_regex_match("I like your graey shirt.", "gr[ae]y")


# In[138]:


# In this example, repeating the character class will match
show_regex_match("I like your graey shirt.", "gr[ae][ae]y")


# In a character class, the `.` character is treated as a literal, not as a wildcard.

# In[139]:


show_regex_match("I like your grey shirt.", "irt[.]")


# There are a few special shorthand notations we can use for commonly used character classes:
# 
# Shorthand | Meaning
# --- | ---
# [0-9] | All the digits
# [a-z] | Lowercase letters
# [A-Z] | Uppercase letters

# In[140]:


show_regex_match("I like your gray shirt.", "y[a-z]y")


# Character classes allow us to create a more specific regex for phone numbers.

# In[141]:


# We replaced every `.` character in ...-...-.... with [0-9] to restrict
# matches to digits.
phone_regex = r'[0-9][0-9][0-9]-[0-9][0-9][0-9]-[0-9][0-9][0-9][0-9]'
show_regex_match("Call me at 382-384-3840.", phone_regex)


# In[142]:


# Now we no longer match this string:
show_regex_match("My truck is not-all-blue.", phone_regex)


# ## Negated Character Classes
# 
# A **negated character class** matches any character **except** the characters in the class. To create a negated character class, wrap the negated characters in `[^ ]`.

# In[143]:


show_regex_match("The car parked in the garage.", r"[^c]ar")


# ## Quantifiers
# 
# To create a regex to match phone numbers, we wrote:
# 
# ```
# [0-9][0-9][0-9]-[0-9][0-9][0-9]-[0-9][0-9][0-9][0-9]
# ```
# 
# This matches 3 digits, a dash, 3 more digits, a dash, and 4 more digits.
# 
# Quantifiers allow us to match multiple consecutive appearances of a pattern. We specify the number of repetitions by placing the number in curly braces `{ }`.

# In[144]:


phone_regex = r'[0-9]{3}-[0-9]{3}-[0-9]{4}'
show_regex_match("Call me at 382-384-3840.", phone_regex)


# In[166]:


# No match
phone_regex = r'[0-9]{3}-[0-9]{3}-[0-9]{4}'
show_regex_match("Call me at 12-384-3840.", phone_regex)


# A quantifier always modifies the character or character class to its immediate left. The following table shows the complete syntax for quantifiers.

# Quantifier | Meaning
# --- | ---
# {m, n} | Match the preceding character m to n times.
# {m} | Match the preceding character exactly m times.
# {m,} | Match the preceding character at least m times.
# {,n} | Match the preceding character at most n times.

# **Shorthand Quantifiers**
# 
# Some commonly used quantifiers have a shorthand:
# 
# Symbol | Quantifier | Meaning
# --- | --- | ---
# * | {0,} | Match the preceding character 0 or more times
# + | {1,} | Match the preceding character 1 or more times
# ? | {0,1} | Match the preceding charcter 0 or 1 times

# We use the `*` character instead of `{0,}` in the following examples.

# In[167]:


# 3 a's
show_regex_match('He screamed "Aaaah!" as the cart took a plunge.', "Aa*h!")


# In[169]:


# Lots of a's
show_regex_match(
    'He screamed "Aaaaaaaaaaaaaaaaaaaah!" as the cart took a plunge.',
    "Aa*h!"
)


# In[151]:


# No lowercase a's
show_regex_match('He screamed "Ah!" as the cart took a plunge.', "Aa*h!")


# **Quantifiers are greedy**
# 
# Quantifiers will always return the longest match possible. This sometimes results in surprising behavior:

# In[173]:


# We tried to match 311 and 911 but matched the ` and ` as well because
# `<311> and <911>` is the longest match possible for `<.+>`.
show_regex_match("Remember the numbers <311> and <911>", "<.+>")


# In many cases, using a more specific character class prevents these false matches:

# In[172]:


show_regex_match("Remember the numbers <311> and <911>", "<[0-9]+>")


# ## Anchoring

# Sometimes a pattern should only match at the beginning or end of a string.  The special character `^` anchors the regex to match only if the pattern appears at the beginning of the string; the special character `$` anchors the regex to match only if the pattern occurs at the end of the string.  For example the regex `well$` only matches an appearance of `well` at the end of the string.

# In[175]:


show_regex_match('well, well, well', r"well$")


# Using both `^` and `$` requires the regex to match the full string.

# In[174]:


phone_regex = r"^[0-9]{3}-[0-9]{3}-[0-9]{4}$"
show_regex_match('382-384-3840', phone_regex)


# In[176]:


# No match
show_regex_match('You can call me at 382-384-3840.', phone_regex)


# ## Escaping Meta Characters

# All regex meta characters have special meaning in a regular expression. To match meta characters as literals, we escape them using the `\` character.

# In[177]:


# `[` is a meta character and requires escaping
show_regex_match("Call me at [382-384-3840].", "\[")


# In[178]:


# `.` is a meta character and requires escaping
show_regex_match("Call me at [382-384-3840].", "\.")


# ## Reference Tables
# 
# We have now covered the most important pieces of regex syntax and meta characters. For a more complete reference, we include the tables below.
# 
# **Meta Characters**
# 
# This table includes most of the important *meta characters*, which help us specify certain patterns we want to match in a string.
# 
# | Char   | Description                         | Example                    | Matches        | Doesn't Match |
# | ------ | ----------------------------------- | -------------------------- | -------------- | ------------- |
# | .      | Any character except \n             | `...`                      | abc            | ab<br>abcd    |
# | [ ]    | Any character inside brackets       | `[cb.]ar`                  | car<br>.ar     | jar           |
# | [^ ]   | Any character _not_ inside brackets | `[^b]ar`                   | car<br>par     | bar<br>ar     |
# | \*     | ≥ 0 or more of last symbol          | `[pb]*ark`                 | bbark<br>ark   | dark          |
# | +      | ≥ 1 or more of last symbol          | `[pb]+ark`                 | bbpark<br>bark | dark<br>ark   |
# | ?      | 0 or 1 of last symbol               | `s?he`                     | she<br>he      | the           |
# | {_n_}  | Exactly _n_ of last symbol          | `hello{3}`                 | hellooo        | hello         |
# | &#124; | Pattern before or after bar         | <code>we&#124;[ui]s</code> | we<br>us<br>is | e<br>s        |
# | \      | Escapes next character              | `\[hi\]`                   | [hi]           | hi            |
# | ^      | Beginning of line                   | `^ark`                     | ark two        | dark          |
# | \$     | End of line                         | `ark$`                     | noahs ark      | noahs arks    |

# **Shorthand Character Sets**
# 
# Some commonly used character sets have shorthands.
# 
# | Description                   | Bracket Form       | Shorthand |
# | ----------------------------- | ------------------ | --------- |
# | Alphanumeric character        | `[a-zA-Z0-9_]`      | `\w`      |
# | Not an alphanumeric character | `[^a-zA-Z0-9_]`     | `\W`      |
# | Digit                         | `[0-9]`            | `\d`      |
# | Not a digit                   | `[^0-9]`           | `\D`      |
# | Whitespace                    | `[\t\n\f\r\p{Z}]`  | `\s`      |
# | Not whitespace                | `[^\t\n\f\r\p{z}]` | `\S`      |

# ## Summary
# 
# Almost all programming languages have a library to match patterns using regular expressions, making them useful regardless of the specific language. In this section, we introduce regex syntax and the most useful meta characters.
