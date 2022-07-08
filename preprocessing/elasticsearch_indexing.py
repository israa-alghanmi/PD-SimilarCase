from elasticsearch import Elasticsearch , RequestsHttpConnection
import argparse
import torch
import logging
import sys
import os
import torch
import numpy as np
import pandas as pd
import sys



def parse_arguments():
    """Read arguments from a command line."""
    parser = argparse.ArgumentParser(description='Experiment setup - Arguments get parsed via --commands')
    parser.add_argument('--data_path', dest='data_path', type=str, default="./wikimed_pubmed_splitted.txt",
        help='The path of the data')
    args = parser.parse_args()
    return args


if __name__ == '__main__':  
    
    if torch.cuda.is_available():        
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    args = parse_arguments()
    

    es=Elasticsearch([{'host':'localhost','port':9200}])
    index_name = 'Wiki_pubMed_splitted'
    file1 = open(args.data_path, 'r')
    Lines = file1.readlines()
    print('len Lines: ', len(Lines), flush=True)    
    index_config = {
    "settings": {
        "analysis": {
            "analyzer": {
                "standard_analyzer": {
                    "type": "standard"
                }
            }
        }
    },
    "mappings": {
      "default_field": { "type": "text"},
        "dynamic": "strict", 
        "properties": {"boolean_sim_field": {"type": "text","similarity": "boolean" } }
        }
    }
    if es.indices.exists(index_name):
        es.indices.delete(index=index_name, ignore=[400, 404])
    
    es.indices.create(index=index_name, body=index_config, ignore=[400, 404])
    print('Creating index done..', flush=True)

    index=0
    for i in range(0,len(Lines)):
        try:
            doc={'text': Lines[i]}
            index_status = es.index(index=index_name, id=index, body=doc)
            index=index+1
            if i %10000 == 0:
                print(i)
                sys.stdout.flush()
        except: 
            break

    print('Indexing Done.......', flush=True)
