#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import warnings


# In[3]:


warnings.filterwarnings('ignore')#to ignore warnings


# ## get the data set

# In[4]:


columns_names=["user_id","item_id","rating","timestamp" ]

df=pd.read_csv("/home/mohit123/Machlea/ml-100k/u.data",sep='\t',names=columns_names)#seperated with \t(tsv)-csv is comma seperated


# In[5]:


df.head()


# In[6]:


get_ipython().run_line_magic('pinfo', 'pd.read_csv')


# In[7]:


df.shape
type(df)


# In[8]:


df["user_id"].nunique()


# In[9]:


df["item_id"].nunique()#no.of unique movies


# In[10]:


movies_title=pd.read_csv("/home/mohit123/Machlea/ml-100k/u.item",sep="\|",header=None) #check the file | is the seperator
#we are reading this file because we want to know which item id corresponds to which movie


# In[11]:


movies_title


# In[12]:


movies_title=movies_title[[0,1]] #we just require movie and item id, i.e first,second col
movies_title.columns=['item_id','title']


# In[13]:


movies_title.head()


# In[14]:


df=pd.merge(df,movies_title,on="item_id")


# In[15]:


df.tail()


# ## exploratory data analysis

# In[16]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')


# In[ ]:


df.mean()


# In[17]:


df.groupby('title').mean()['rating'].sort_values(ascending=False)#we are concerned by average rating of movie


# In[18]:


##very rarely a movie can get an average of 5 star.Therefore if someone gaets 5 rating means probably less people watched that (avoid such movies)


# In[19]:


df.groupby('title').count()['rating'].sort_values(ascending=False)


# In[20]:


ratings=pd.DataFrame(df.groupby('title').mean()['rating'])


# In[21]:


type(ratings)


# In[22]:


ratings.head()


# In[23]:


ratings['num of ratings']=pd.DataFrame(df.groupby('title').count()['rating'])


# In[24]:


ratings


# In[25]:


ratings.sort_values(by='rating',ascending=False)


# In[26]:


plt.figure(figsize=(10,6))
plt.hist(ratings['num of ratings'],bins=70)
plt.show()


# In[27]:


plt.hist(ratings['rating'],bins=70)
plt.show()


# In[28]:


get_ipython().run_line_magic('pinfo', 'sns.jointplot')


# In[32]:


sns.jointplot(x='rating',y='num of ratings',data=ratings,alpha=0.5)


# ## Creating Movie Recommendation system

# In[33]:


df.head()


# In[35]:


moviemat=df.pivot_table(index="user_id",columns="title",values="rating")


# In[36]:


moviemat


# In[37]:


ratings.sort_values('num of ratings',ascending=False).head()


# In[38]:


starwars_user_ratings=moviemat['Star Wars (1977)']


# In[39]:


starwars_user_ratings.head()


# In[65]:


similar_to_starwars=moviemat.corrwith(starwars_user_ratings)
type(similar_to_starwars)


# In[52]:


corr_starwars=pd.DataFrame(similar_to_starwars,columns=['Correlation'])


# In[53]:


corr_starwars.dropna()


# In[54]:


corr_starwars


# In[55]:


corr_starwars.dropna(inplace=True)


# In[56]:


corr_starwars


# In[57]:


corr_starwars.head()


# In[58]:


corr_starwars.sort_values('Correlation',ascending=False).head(10)


# In[59]:


#if a movie is rated by 100 users then only suggest that movie


# In[60]:


corr_starwars


# In[61]:


ratings


# In[63]:


corr_starwars=corr_starwars.join(ratings['num of ratings'])
corr_starwars.head()


# In[64]:


corr_starwars[corr_starwars['num of ratings']>100].sort_values('Correlation',ascending=False)


# ## predict function

# In[69]:


def predict_movies(movie_name):
    movie_user_ratings=moviemat[movie_name]
    similar_to_movie=moviemat.corrwith(movie_user_ratings)
    
    corr_movie=pd.DataFrame(similar_to_movie,columns=['Correlation'])
    corr_movie.dropna(inplace=True)
    
    corr_movie=corr_movie.join(ratings['num of ratings'])
    predictions=corr_movie[corr_movie['num of ratings']>100].sort_values('Correlation',ascending=False)
    
    return predictions


# In[70]:


predictions= predict_movies("Titanic (1997)")


# In[71]:


predictions.head()


# In[ ]:




