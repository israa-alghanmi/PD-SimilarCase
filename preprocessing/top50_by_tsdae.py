# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 17:20:05 2020

@author: Israa
"""

import pandas as pd
import numpy as np
import pickle
import spacy
from collections import OrderedDict
from pathlib import Path
import sys
import argparse
import json
import torch
from elasticsearch import Elasticsearch 
from sentence_transformers import SentenceTransformer, util
from transformers import *
import torch.nn as nn
import os
import random
import gc 
import re

def parse_arguments():
    """Read arguments from a command line."""
    parser = argparse.ArgumentParser(description='Experiment setup - Arguments get parsed via --commands')
    parser.add_argument('--train', dest='training_path', type=str, default="", help='')
    parser.add_argument('--dev', dest='dev_path', type=str, default="",help='')
    parser.add_argument('--testing', dest='testing_path', type=str, default="",help=' ')
    parser.add_argument('--TSDAE_path', dest='TSDAE_path', type=str, default="",help='')
    parser.add_argument('--index_name', dest='index_name', type=str, default="",help='')
    args = parser.parse_args()
    return args
  



def add_context_column(df,es):
    df = df.replace(np.nan, '', regex=True)
    df['context_wiki_pub']=''
    model_path=args.TSDAE_path
    Sentence_model = SentenceTransformer(model_path)
    for i in range (0,len(df)):
        if i % 100 == 0:
            print(i, flush=True)
        question=df['question'][i]
        option=df['option'][i]
        new_sring=''
        new_list=[]
        #. encode question
        question_embeddings = Sentence_model.encode(question, convert_to_tensor=True)
        try: 
            res= es.search(index=args.index_name,size=1000,body={'query':{'match_phrase':{"text":option}}})
            context_sentences=[]
            #Compute cosine-similarits
            cosine_scores=[]
            for hit in res['hits']['hits']:
                Answer_context_sentence_embeddings = Sentence_model.encode(hit['_source']['text'] , convert_to_tensor=True) 
                cosine_scores.append(util.pytorch_cos_sim(question_embeddings, Answer_context_sentence_embeddings))
                context_sentences.append(hit['_source']['text'])            

            if len(cosine_scores)> 0:
                new_tuple=sorted(zip(cosine_scores, context_sentences), reverse=True)
                new_list=[y for x,y in new_tuple[:50]]
                new_sring='\t\t'.join(map(str,new_list))

        except: 
            print(i,'None')
            continue    
        df['context_wiki_pub'][i]=new_sring
            
    return df 
    
        
  


              
if __name__ == '__main__': 
    args = parse_arguments() 
    pd.options.mode.chained_assignment = None
    es=Elasticsearch([{'host':'localhost','port':9200}])

    training_df= pd.read_csv(args.training_path)  
    dev_df= pd.read_csv(args.dev_path)  
    testing_df= pd.read_csv(args.testing_path)      

    training_df=add_context_column(training_df,es)
    dev_df=add_context_column(dev_df,es)
    testing_df=add_context_column(testing_df,es)
    training_df.to_csv("./MedQA_training_with_top50.csv")
    dev_df.to_csv("./MedQA_dev_with_top50.csv")    
    testing_df.to_csv("./MedQA_test_with_top50.csv")   


        