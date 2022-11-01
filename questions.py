import numpy as np 
import pandas as pd 
import collections
import os

ques = pd.read_csv('Questions.csv',encoding='iso-8859-1')
ques.head(10)

ques.drop(["OwnerUserId","CreationDate","ClosedDate","Score"], axis=1, inplace=True)
ques.head(10)

import re 

def rem_html_tags(body):
    regex = re.compile('<.*?>')
    return re.sub(regex, '', body)

ques['Body'] = ques['Body'].apply(rem_html_tags)
ques.head()

ques.to_csv('question_clean.csv',index=False)
