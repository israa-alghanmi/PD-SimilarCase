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

def parse_arguments():
    """Read arguments from a command line."""
    parser = argparse.ArgumentParser(description='Experiment setup - Arguments get parsed via --commands')
    parser.add_argument('--training', dest='training', type=str, default="", help='')
    parser.add_argument('--dev', dest='dev', type=str, default="", help='')
    parser.add_argument('--test', dest='test', type=str, default="", help='')
    args = parser.parse_args()
    return args


def load_jsonl(input_path) -> list:
    """
    Read list of objects from a JSON lines file.
    """
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.rstrip('\n|\r')))
    print('Loaded {} records from {}'.format(len(data), input_path))
    return data

def get_df_from_json(data):
    question=[]
    options=[]
    answer_idx=[]
    answer=[] 
    A=[]
    B=[]
    C=[]
    D=[]
    for i in range (0,len(data)):
            question.append(data[i]['question'])
            answer.append(data[i]['answer'])
            answer_idx.append(data[i]['answer_idx'])
            options.append(data[i]['options'])
            #for ans in options:
            A.append(data[i]['options']['A'])
            B.append(data[i]['options']['B'])
            C.append(data[i]['options']['C'])
            D.append(data[i]['options']['D'])
                 
    d = {'question':question,'options':options, 'A':A, 'B':B, 'C':C,'D':D,'answer_idx':answer_idx,'answer':answer }
    df = pd.DataFrame.from_dict(d)
    return df



def get_new_df(df):
    questions=[]
    options=[]
    labels=[]
    
    for i in range(0,len(df)):
        question=df['question'][i]
        answer_a=df['A'][i]
        answer_b=df['B'][i]
        answer_c=df['C'][i]
        answer_d=df['D'][i]  
        answers=[answer_a,answer_b,answer_c,answer_d]
        
        correct_answer=df[df['answer_idx'][i]][i]
        
        for answer in answers:
            questions.append(question)
            options.append(answer)
            if answer==correct_answer:
                labels.append(1)
            else: labels.append(0)
                
    data = {'question':questions,'option':options, 'label':labels }

    # Create DataFrame
    new_df = pd.DataFrame(data)
    
    return new_df
    
    
if __name__ == '__main__':  
    
  if torch.cuda.is_available():        
      device = torch.device("cuda")
  else:
      device = torch.device("cpu")
  args = parse_arguments()
  
  training_data= load_jsonl(args.training)
  dev_data= load_jsonl(args.dev)
  test_data= load_jsonl(args.test)
  
  training_df= get_new_df(get_df_from_json(training_data))
  dev_df= get_new_df(get_df_from_json(dev_data))
  test_df= get_new_df(get_df_from_json(test_data))

  
  training_df.to_csv('./MedQA_training.csv', index=True) 
  dev_df.to_csv('./MedQA_dev.csv', index=True) 
  test_df.to_csv('./MedQA_test.csv', index=True) 