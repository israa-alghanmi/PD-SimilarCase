# -*- coding: utf-8 -*-

import sys
import argparse
import json
import pandas as pd
from transformers import *
from sklearn.metrics import classification_report,f1_score,accuracy_score
import numpy as np
import torch
import time
import datetime
import random
from sklearn.utils import shuffle
import gc 
from sklearn.metrics import average_precision_score
import torch.nn as nn
import torch.nn.functional as fnn
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator,CESoftmaxAccuracyEvaluator,CECorrelationEvaluator
from sentence_transformers.readers import InputExample
import math
from sentence_transformers import SentenceTransformer, util
from random import randrange

def parse_arguments():
    """Read arguments from a command line."""
    parser = argparse.ArgumentParser(description='Experiment setup - Arguments get parsed via --commands')
    parser.add_argument('--train', dest='training_path', type=str, default="",help='')
    parser.add_argument('--dev', dest='dev_path', type=str, default="",help='')
    parser.add_argument('--testing', dest='testing_path', type=str, default="",help='')
    parser.add_argument('--seed_val', dest='seed_val', type=int, default="42", help=')
    parser.add_argument('--pretrained_cross_encoder_path', dest='cross_encoder_path', type=str, default="", help='cross_encoder_path ')
    parser.add_argument('--top_k', dest='agum_size', type=int, default=1, help='value of K')
    parser.add_argument('--temp_dir', dest='temp_dir', type=str, default="", help='temporary directory')
    parser.add_argument('--model_path', dest='model_path', type=str, default="", help='BERT path')
    args = parser.parse_args()
    return args
    


def read_csv_to_df(input_path):
    df= pd.read_csv(input_path)
    df.label= df.label.apply(lambda x: int(x))
    return df
    


def train(train_samples,dev_samples,evaluator,num_labels,path):


    ### Hyperparameters Tuning ###

    train_batch_size = 8
    num_epochs = 4
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
    model = CrossEncoder(path, num_labels=num_labels,max_length=512)

    # Configure the training
    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
    
    # Train the model
    model.fit(train_dataloader=train_dataloader,
              evaluator=evaluator,
              epochs=num_epochs,
              evaluation_steps=5000,
              warmup_steps=warmup_steps) 
              
    return model


    
def test_mAP(sentence_model,df, dt,tokenizer):
    labels=[]
    predictions=[]
    final_context_sentences =[]
    df = df.replace(np.nan, '', regex=True)
    Question_scores=[] 
    predictions_per_question=[] 
    Question_scores_by_index=[]
    predictions_by_index=[] 
    temp_labels=[]
    labels_by_index=[]
    for i in range(0,len(df)):    
        label= df['label'][i]
        final_context_sentences=df['context_wiki_pub'][i].split('\t\t') 
        if len(final_context_sentences)<2 and final_context_sentences[0]=='':
            final_context_sentences=[]
            final_context_sentences.append(df['option'][i])
        pairs=[[df['question'][i], context] for context in final_context_sentences]
        scores = sentence_model.predict(pairs)                  
        scores= scores.tolist()
        predictions.append(max(scores))
        labels.append(label)
        temp_labels.append(label)
        Question_scores.append(max(scores))
        Question_scores_by_index.append(max(scores))
        if len(Question_scores)==4:
            Question_scores=[1 if score == max(Question_scores) else 0 for score in Question_scores] 
            predictions_per_question=predictions_per_question+Question_scores
            Question_scores=[]
            Question_scores_by_index=Question_scores_by_index.index(max(Question_scores_by_index))
            predictions_by_index.append(Question_scores_by_index)
            Question_scores_by_index=[]
            temp_labels=temp_labels.index(1)
            labels_by_index.append(temp_labels)
            temp_labels=[]
                       
    print("Accuracy:{}".format(accuracy_score(labels_by_index, predictions_by_index)))
    
    return labels, predictions  
    
def get_samples(df):
    samples = []
    df=df.reset_index(drop=True)
    for i in range(0,len(df)):
        l=int(df['label'][i])
        samples.append(InputExample(texts=[str(df['question'][i]), str(df['option'][i])], label=l))
    return samples
    


def get_samples_with_context(df,model_path_org,agum_size):
    samples = []
    df = df.replace(np.nan, '', regex=True) 
    model_path2=model_path_org
    sentence_model2 = CrossEncoder(model_path2,max_length=512)
    tokenizer2 = AutoTokenizer.from_pretrained(model_path2)
    df=df.reset_index(drop=True)
    for i in range(0,len(df)):
        final_context_sentences=str(df['context_wiki_pub'][i]).split('\t\t') 
        if len(final_context_sentences)<2 and final_context_sentences[0]=='':
            final_context_sentences=[]
            final_context_sentences.append(df['option'][i])
        best_context=''  
        pairs=[[df['question'][i], context] for context in final_context_sentences]
        scores = sentence_model2.predict(pairs)
        scores= scores.tolist()
        predicted_context= get_predicted_context(scores, final_context_sentences) 
        l=int(df['label'][i])
        samples.append(InputExample(texts=[str(df['question'][i]), predicted_context], label=l))
        scores2,final_context_sentences=remove_max_score_context(scores,final_context_sentences)
        for j in range(1, int(agum_size)):
            if len(final_context_sentences)>0:
                predicted_context= get_predicted_context(scores2, final_context_sentences)
                samples.append(InputExample(texts=[str(df['question'][i]), predicted_context], label=l))
                scores2,final_context_sentences=remove_max_score_context(scores2,final_context_sentences)
    del sentence_model2
    print('samples len: ', len(samples))
    return samples 
        


    
def get_predicted_context(scores, final_context_sentences):
    max_index=scores.index(max(scores))
    predicted_context= final_context_sentences[max_index]
    return predicted_context

        
def remove_max_score_context(scores, final_context_sentences):
    scores2=scores.copy()
    largest_score = max(scores2) 
    max_index=scores2.index(max(scores2))
    scores2.remove(largest_score)
    final_context_sentences.remove(final_context_sentences[max_index])

    return scores2, final_context_sentences
    



if __name__ == '__main__':  
    
    if torch.cuda.is_available():        
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    args = parse_arguments()
    ## Hyperparameters ## 
    lr= 2e-5
    epochs = 4
    seed_val=args.seed_val
    batch_size=8

 
    training_df= read_csv_to_df(args.training_path)
    dev_df= read_csv_to_df(args.dev_path)
    test_df= read_csv_to_df(args.testing_path)

    BERT_path=args.model_path
    model_path_org=args.cross_encoder_path
    model_c = AutoModel.from_pretrained(model_path_org)
    directory = str(randrange(97654235))
    # Parent Directory path
    parent_dir = args.temp_dir
    # Path
    model_path = os.path.join(parent_dir, directory)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    model_c.save_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path_org)
    tokenizer.save_pretrained(model_path)
    agum_size=args.agum_size   
    print('----Top:',agum_size)    
    training_samples=get_samples_with_context(training_df,model_path_org,agum_size)
    dev_samples=get_samples_with_context(dev_df,model_path_org,agum_size)
    evaluator= CEBinaryClassificationEvaluator.from_input_examples(dev_samples, name='dev') 
    sentence_model= train(training_samples,dev_samples,evaluator,1,model_path)
    y_test, predictions = test_mAP(sentence_model,test_df, 'test' ,tokenizer)
    
    print('<--------------- AP ----------------->')   
    print("Average :{}".format(average_precision_score(y_test, predictions)))
    print('##########################################')    

    gc.collect()
    torch.cuda.empty_cache()
          
      