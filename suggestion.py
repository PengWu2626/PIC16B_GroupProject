#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np


# In[6]:


df=pd.read_csv("/Users/yuanyuansu/desktop/results.csv")


# In[7]:


df


# In[32]:


def found_best_suggestion(df):
    do_not_want=input('What is the character of a dog you do not like at all:')
    first_choice=input('What is the most important character of a dog for you:')
    df0=df.loc[df['characteristic']==do_not_want]
    df00=df0[df0['star']>=3]
    L0=df00["breed"].tolist()
    df1=df[df['characteristic']==first_choice]
    df11=df1[df1['star']>=4]
    L1=df11["breed"].tolist()
    best_suggestion=list(set(L1) - set(L0))
    best_suggestion
    if len(best_suggestion)==0:
        ans=input("There is no good suggestion for your preference. Do you want to try some other characters?(Y/N)")
        if ans=='Y':
            found_best_suggestion(df)
        else:
            print('Sorry to hear that.')
    elif len(best_suggestion)>3:
        second_choice=input('What is the next character of a dog you like:')
        df2=df[df['characteristic']==second_choice]
        df22=df2[df2['star']>=4]
        L2=df22["breed"].tolist()   
        L12=list(set.intersection(set(L1), set(L2)))
        best_suggestion1=list(set(L12) - set(L0))
        if len(best_suggestion1)>3:
            third_choice=input('What is the next character of a dog you like:')
            df3=df[df['characteristic']==third_choice]
            df33=df3[df3['star']>=3]
            L3=df33["breed"].tolist()
            suggestion_3=list(set.intersection(set(L1), set(L3)))
            suggestion_4=list(set.intersection(set(L2), set(L3)))
            suggestion2=list(set.intersection(set(L12), set(third_list)))
            suggestion=L12+suggestion_3+suggestion_4
            best_suggestion2=list(set(L12) - set(L0))
            if len(best_suggestion2)==0:
                print('There is no suggestion for'+ third_choice +'but the suggestion for other two characters is'+best_suggestion[0:3])
            elif len(best_suggestion2)>3:
                print(best_suggestion2[0:3])
            else:
                print(best_suggestion2)
        elif len(best_suggestion1)==0:    
            print('There is no suggestion for'+ second_choice +'character,but the suggestion for other two characters is'+best_suggestion[0:3])
        else:
            print(best_suggestion1)
    else:
        print(best_suggestion)


# In[33]:


found_best_suggestion(df)

