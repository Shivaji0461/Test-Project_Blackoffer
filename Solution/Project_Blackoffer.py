#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


# !pip install contractions


# In[3]:


# !pip install newspaper3k


# In[4]:


# Import Essential libraries

import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
import contractions
from newspaper import Article
import nltk
from collections import  Counter
from textblob import TextBlob
import numpy as np


# In[5]:


# Import the URL dataset

# pd.set_option('display.max_colwidth',200)
data = pd.read_excel('Input.xlsx')
data.head()


# In[6]:


# Removed 3nos. url because they have found error
data = data.loc[data.index.drop([7,20,107])]
data = data.reset_index(drop=True)
data.head()


# In[7]:


# Extract the article text and article title.
def get_text(url):
    url1 = url
    article = Article(url1,language='en')
    article.download()
    article.parse()
    article.nlp()
    return article.text


# In[8]:


for i in range(len(data)):
    data['URL_ID'][i] = get_text(data['URL'][i])

data.head()


# In[9]:


data = data.rename(columns={'URL_ID':'text'})
data.head()


# # 1. Sentimental Analysis

# # 1.1. Cleaning using Stop Words Lists

# In[10]:


# Import the stopword dataset

stop_1 = pd.read_csv('StopWords_Auditor.txt',header=None)
stop_2 = pd.read_csv('StopWords_Currencies.txt',header=None,sep='delimeter',encoding="ISO-8859-1")
stop_3 = pd.read_csv('StopWords_DatesandNumbers.txt',header=None)
stop_4 = pd.read_csv('StopWords_Generic.txt',header=None)
stop_5 = pd.read_csv('StopWords_GenericLong.txt',header=None)
stop_6 = pd.read_csv('StopWords_Geographic.txt',header=None)
stop_7 = pd.read_csv('StopWords_Names.txt',header=None)

stop_words = pd.concat([stop_1,stop_2,stop_3,stop_4,stop_5,stop_6,stop_7])
stop_words = stop_words.iloc[:, 0].tolist()

print(stop_words)


# In[11]:


# Text preprocessing stopword dataset

remove_url = re.sub('[^ ]+\.[^ ]+',' ', str(stop_words))

review_contraction = []   
for word in remove_url:
    (review_contraction.append(contractions.fix(word)))
review_contraction = ''.join(review_contraction)

text_preprocess = re.sub(r'[^\w\s]','',review_contraction)
digits = ''.join([re.sub('\S*\d\S*',' ',term) for term in text_preprocess])
extra_spaces = re.sub('\s\s+',' ',digits)
stop_words = [token.lower() for token in word_tokenize(digits)]

print(stop_words)


# In[12]:


# Preprocessing on article text dataset
def Text_Preprocessing(text):
    content = re.sub(r'[^\w\s]','',text)
    digits = ''.join([re.sub('\S*\d\S*',' ',term) for term in content])
    extra_spaces = re.sub('\s\s+',' ',digits)
    tokens = [token.lower() for token in word_tokenize(extra_spaces)]
    non_stopwords = [word for word in tokens if word not in stop_words]
    return non_stopwords


# In[13]:


data['preprocess_text'] = data['text'].apply(Text_Preprocessing)
data.head()


# ## 1.2. Creating a dictionary of Positive and Negative words

# In[14]:


with open('positive-words.txt') as pos:
    posswords = pos.read().split('\n')
posswords = ' ' .join(posswords)


# In[15]:


with open('negative-words.txt',encoding="ISO-8859-1") as neg:
        negwords = neg.read().split('\n')


# ## 1.3. Extracting Derived variables

# ### 1. POSITIVE SCORE

# In[16]:


def positive_text(text):
  pos_count = ' '.join([word for word in text if word in posswords])
  pos_count = pos_count.split(' ')
  positive_score = len(pos_count)
  return positive_score


# In[17]:


data['POSITIVE SCORE'] = data['preprocess_text'].apply(positive_text)
data.head()


# ### 2. NEGATIVE SCORE

# In[18]:


def negative_text(text):
  neg_count = ' '.join([word for word in text if word in negwords])
  neg_count = neg_count.split(' ')
  positive_score = len(neg_count)
  return positive_score


# In[19]:


data['NEGATIVE SCORE'] = data['preprocess_text'].apply(negative_text)
data.head()


# ### 3. POLARITY SCORE

# This is the score that determines if a given text is positive or negative in nature. It is calculated by using the formula: 
# Polarity Score = (Positive Score – Negative Score)/ ((Positive Score + Negative Score) + 0.000001)
# 
# Range is from -1 to +1

# In[20]:


def polarity(text):
  sentiment = TextBlob(text).sentiment
  polarity = sentiment.polarity
  return polarity


# In[21]:


data['POLARITY SCORE'] = np.nan

for i in range(len(data)):
    data['POLARITY SCORE'][i] = (data['POSITIVE SCORE'][i]-data['NEGATIVE SCORE'][i])/ ((data['POSITIVE SCORE'][i] + data['NEGATIVE SCORE'][i]) + 0.000001)


# In[22]:


data.head()


# ### 4. SUBJECTIVITY SCORE

# This is the score that determines if a given text is objective or subjective. It is calculated by using TextBlob library.
# 
# Range is from 0 to +1

# In[23]:


data['SUBJECTIVITY SCORE'] = np.nan

for i in range(len(data)):
    data['SUBJECTIVITY SCORE'][i] = TextBlob(str(data['preprocess_text'][i])).sentiment[1]
data.head()


# In[24]:


data['WORD COUNT'] = data['preprocess_text'].apply(lambda x: len(x))
data.head()


# # 2. Analysis of Readability

# Analysis of Readability is calculated using the Gunning Fox index formula described below.
# 
# Average Sentence Length = the number of words / the number of sentences
# 
# Percentage of Complex words = the number of complex words / the number of words 
# 
# Fog Index = 0.4 * (Average Sentence Length + Percentage of Complex words)

# In[25]:


data['AVG SENTENCE LENGTH'] = np.nan
for i in range(len(data)):
  data['AVG SENTENCE LENGTH'][i] = data['WORD COUNT'][i] / len(sent_tokenize(str(data['text'][i])))


# In[26]:


data.head()


# # 3. Average Number of Words Per Sentence

# The formula for calculating is:
#     
# Average Number of Words Per Sentence = the total number of words / the total number of sentences

# In[27]:


data['AVG NUMBER OF WORDS PER SENTENCE'] = np.nan
for i in range(len(data)):
  data['AVG NUMBER OF WORDS PER SENTENCE'][i] = round(data['WORD COUNT'][i] / len(sent_tokenize(str(data['text'][i]))),2) 


# In[28]:


data.head()


# # 4. Complex Word Count
# Complex words are words in the text that contain more than two syllables.
# 

# In[29]:


def complex_word_count(x):
    
    syllable = 'aeiou'
    
    t = x
    
    v = []
    
    for i in t:
        words = i
        c=Counter()
        
        for word in words:
            c.update(set(word))

        n = 0
        for a in c.most_common():
            if a[0] in syllable:
                if a[1] >= 2:
                    n += 1
                
        m = 0
        p = []
        for a in c.most_common():
            if a[0] in syllable:
                p.append(a[0])
        if len(p) >= 2:
            m += 1
        
        if n >= 1 or m >= 1:
            v.append(i)
            
    return len(v) 


# In[30]:


data['COMPLEX WORD COUNT'] = np.nan

data['COMPLEX WORD COUNT'] = data['preprocess_text'].apply(lambda x: complex_word_count(x))
data.head()


# In[31]:


data['PERCENTAGE OF COMPLEX WORDS'] = np.nan

for i in range(len(data)):
  data['PERCENTAGE OF COMPLEX WORDS'][i] = data['COMPLEX WORD COUNT'][i] / data['WORD COUNT'][i]
data.head()


# In[32]:


data['FOG INDEX'] = np.nan

for i in range(len(data)):
  data['FOG INDEX'][i] = 0.4 * (data['AVG NUMBER OF WORDS PER SENTENCE'][i] + data['PERCENTAGE OF COMPLEX WORDS'][i])
data.head()


# # 6 Syllable Count Per Word
# We count the number of Syllables in each word of the text by counting the vowels present in each word. We also handle some exceptions like words ending with "es","ed" by not counting them as a syllable.

# In[33]:


def syllable_count(word):
    word = word.lower()
    count = 0
    vowels = "aeiouy"
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith("e"):
        count -= 1
    if count == 0:
        count += 1
    return count


# In[34]:


data['SYLLABLE PER WORD'] = np.nan
for i in range(len(data)):
  data['SYLLABLE PER WORD'][i] = syllable_count(''.join(data['preprocess_text'][i]))

data.head()


# # 7. Personal Pronouns
# To calculate Personal Pronouns mentioned in the text, we use regex to find the counts of the words - “I,” “we,” “my,” “ours,” and “us”. Special care is taken so that the country name US is not included in the list.
# 

# In[35]:


def ProperNoun(text):
    count = 0
    text = ' '.join(text)
    sentences = sent_tokenize(text)
    for sentence in sentences:
        words = word_tokenize(sentence)
        tagged = nltk.pos_tag(words)
        for word, tag in tagged:
            if tag == 'PRP':
                count = count + 1
        return count


# In[36]:


data['PERSONAL PRONOUNS'] = data['preprocess_text'].apply(ProperNoun)


# In[37]:


data.head()


# # 8. Average Word Length
# Average Word Length is calculated by the formula:
#     
# Sum of the total number of characters in each word/Total number of words
# 

# In[38]:


data['AVG WORD LENGTH'] =  np.nan

for i in range(len(data)):
  data['AVG WORD LENGTH'][i] = round(len(''.join(data['preprocess_text'][i])) / data['WORD COUNT'][i],2)

data.head()


# In[39]:


df = data[['URL','POSITIVE SCORE','NEGATIVE SCORE','POLARITY SCORE','SUBJECTIVITY SCORE','AVG SENTENCE LENGTH','PERCENTAGE OF COMPLEX WORDS','FOG INDEX',
           'AVG NUMBER OF WORDS PER SENTENCE','COMPLEX WORD COUNT','WORD COUNT','SYLLABLE PER WORD','PERSONAL PRONOUNS','AVG WORD LENGTH']]


# In[40]:


df.index = df.index+37


# In[41]:


df.head()


# In[42]:


df.to_excel("Output Data Structure.xlsx")


# In[ ]:




