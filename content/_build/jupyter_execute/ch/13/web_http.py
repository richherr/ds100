#!/usr/bin/env python
# coding: utf-8

# # HTTP
# 
# HTTP (AKA **H**yper**T**ext **T**ransfer **P**rotocol) is a *request-response* protocol that allows one computer to talk to another over the Internet.

# ## Requests and Responses
# 
# The Internet allows computers to send text to one another, but does not impose any restrictions on what that text contains. HTTP defines a structure on the text communication between one computer (client) and another (server). In this protocol, a client submits a *request* to a server, a specially formatted text message. The server sends a text *response* back to the client.
# 
# The command line tool `curl` gives us a simple way to send HTTP requests. In the output below, lines starting with `>` indicate the text sent in our request; the remaining lines are the server's response.

# ```bash
# $ curl -v https://httpbin.org/html
# ```

# ```
# > GET /html HTTP/1.1
# > Host: httpbin.org
# > User-Agent: curl/7.55.1
# > Accept: */*
# > 
# < HTTP/1.1 200 OK
# < Connection: keep-alive
# < Server: meinheld/0.6.1
# < Date: Wed, 11 Apr 2018 18:15:03 GMT
# < 
# <html>
#   <body>
#     <h1>Herman Melville - Moby-Dick</h1>
#     <p>
#       Availing himself of the mild...
#     </p>
#   </body>
# </html>
# ```

# Running the `curl` command above causes the client's computer to construct a text message that looks like:
# 
# ```
# GET /html HTTP/1.1
# Host: httpbin.org
# User-Agent: curl/7.55.1
# Accept: */*
# {blank_line}
# ```
# 
# This message follows a specific format: it starts with `GET /html HTTP/1.1` which indicates that the message is an HTTP `GET` request to the `/html` page. Each of the three lines that follow form HTTP headers, optional information that `curl` sends to the server. The HTTP headers have the format `{name}: {value}`. Finally, the blank line at the end of the message tells the server that the message ends after three headers. Note that we've marked the blank line with `{blank_line}` in the snippet above; in the actual message `{blank_line}` is replaced with a blank line.

# The client's computer then uses the Internet to send this message to the `https://httpbin.org` web server. The server processes the request, and sends the following response:
# 
# ```
# HTTP/1.1 200 OK
# Connection: keep-alive
# Server: meinheld/0.6.1
# Date: Wed, 11 Apr 2018 18:15:03 GMT
# {blank_line}
# ```
# 
# The first line of the response states that the request completed successfully. The following three lines form the HTTP response headers, optional information that the server sends back to the client. Finally, the blank line at the end of the message tells the client that the server has finished sending its response headers and will next send the response body:
# 
# ```
# <html>
#   <body>
#     <h1>Herman Melville - Moby-Dick</h1>
#     <p>
#       Availing himself of the mild...
#     </p>
#   </body>
# </html>
# ```

# This HTTP protocol is used in almost every application that interacts with the Internet. For example, visiting https://httpbin.org/html in your web browser makes the same basic HTTP request as the `curl` command above. Instead of displaying the response as plain text as we have above, your browser recognizes that the text is an HTML document and will display it accordingly.
# 
# In practice, we will not write out full HTTP requests in text. Instead, we use tools like `curl` or Python libraries to construct requests for us.

# ## In Python

# The Python **requests** library allows us to make HTTP requests in Python. The code below makes the same HTTP request as running `curl -v https://httpbin.org/html`.

# In[1]:


import requests

url = "https://httpbin.org/html"
response = requests.get(url)
response


# ### The Request

# Let's take a closer look at the request we made. We can access the original request using `response` object; we display the request's HTTP headers below:

# In[2]:


request = response.request
for key in request.headers: # The headers in the response are stored as a dictionary.
    print(f'{key}: {request.headers[key]}')


# Every HTTP request has a type. In this case, we used a `GET` request which retrieves information from a server.

# In[3]:


request.method


# ### The Response
# 
# Let's examine the response we received from the server. First, we will print the response's HTTP headers.

# In[4]:


for key in response.headers:
    print(f'{key}: {response.headers[key]}')


# An HTTP response contains a status code, a special number that indicates whether the request succeeded or failed. The status code `200` indicates that the request succeeded.

# In[5]:


response.status_code


# Finally, we display the first 100 characters of the response's content (the entire response content is too long to display nicely here).

# In[6]:


response.text[:100]


# ## Types of Requests
# 
# The request we made above was a `GET` HTTP request. There are multiple HTTP request types; the most important two are `GET` and `POST` requests.

# ### GET Requests
# 
# The `GET` request is used to retrieve information from the server. Since your web browser makes `GET` request whenever you enter in a URL into its address bar, `GET` requests are the most common type of HTTP requests.
# 
# `curl` uses `GET` requests by default, so running `curl https://www.google.com/` makes a `GET` request to `https://www.google.com/`.

# ### POST Request
# 
# The `POST` request is used to send information from the client to the server. For example, some web pages contain forms for the user to fill out???a login form, for example. After clicking the "Submit" button, most web browsers will make a `POST` request to send the form data to the server for processing.

# Let's look an example of a `POST` request that sends `'sam'` as the parameter `'name'`. This one can be done by running **`curl -d 'name=sam' https://httpbin.org/post`** on the command line. 
# 
# Notice that our request has a body this time (filled with the parameters of the `POST` request), and the content of the response is different from our `GET` response from before. 

# Like HTTP headers, the data sent in a `POST` request uses a key-value format. In Python, we can make a `POST` request by using `requests.post` and passing in a dictionary as an argument.

# In[7]:


post_response = requests.post("https://httpbin.org/post",
                              data={'name': 'sam'})
post_response


# The server will respond with a status code to indicate whether the `POST` request successfully completed. In addition, the server will usually send a response body to display to the client.

# In[8]:


post_response.status_code


# In[9]:


post_response.text


# ## Types of Response Status Codes
# 
# The previous HTTP responses had the HTTP status code `200`. This status code indicates that the request completed successfully. There are hundreds of other HTTP status codes. Thankfully, they are grouped into categories to make them easier to remember:
# 
# - **100s** - Informational: More input is expected from client or server *(e.g. 100 Continue, 102 Processing)*
# - **200s** - Success: The client's request was successful *(e.g. 200 OK, 202 Accepted)*
# - **300s** - Redirection: Requested URL is located elsewhere; May need user's further action *(e.g. 300 Multiple Choices, 301 Moved Permanently)*
# - **400s** - Client Error: Client-side error *(e.g. 400 Bad Request, 403 Forbidden, 404 Not Found)*
# - **500s** - Server Error: Server-side error or server is incapable of performing the request *(e.g. 500 Internal Server Error, 503 Service Unavailable)*
# 
# We can look at examples of some of these errors.

# In[10]:


# This page doesn't exist, so we get a 404 page not found error
url = "https://www.youtube.com/404errorwow"
errorResponse = requests.get(url)
print(errorResponse)


# In[11]:


# This specific page results in a 500 server error
url = "https://httpstat.us/500"
serverResponse = requests.get(url)
print(serverResponse)


# ## Summary
# 
# We have introduced the HTTP protocol, the basic communication method for applications that use the Web. Although the protocol specifies a specific text format, we typically turn to other tools to make HTTP requests for us, such as the command line tool `curl` and the Python library `requests`.
