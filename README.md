# Distantly Supervised Similar Patient Case Retrieval
This repository contains the official implementation of the paper [Interpreting Patient Descriptions using Distantly Supervised Similar Case Retrieval](add_link_here) which has been accepted by the SIGIR 2022 conference.

#### Requirements
1. Text Corpus
* [WikiMed and PubMedDS](https://www.sciencedirect.com/science/article/pii/S1532046421002094) 
* [MIMIC-III discharge summaries](https://physionet.org/content/mimiciii/1.4/)
2. TSDAE Model 
* We train our TSDAE model using this implementation [train_tsdae_from_file.py](https://github.com/UKPLab/sentence-transformers/blob/master/examples/unsupervised_learning/TSDAE/train_tsdae_from_file.py), we initialize the model from ClinicalBERT, and train on MIMIC-III discharge summaries splitted into text passages
3. Pre-training Datasets
* [STS-B](https://aclanthology.org/S17-2001.pdf?ref=https://githubhelp.com)
* [RQE](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5333286/)
* [HealthQA](https://dl.acm.org/doi/abs/10.1145/3308558.3313699)
4. Evaluation Datasets
* [MedQA](https://www.mdpi.com/2076-3417/11/14/6421/htm)
* [DisKnE](https://aclanthology.org/2021.findings-acl.266.pdf)
* [HeadQA](https://aclanthology.org/P19-1092.pdf)
## Pre-processing
We convert the multiple-choice question answering datasets to binary classification. 
Here is an example for MedQA:
```
python3 "preprocessing/preprocessing.py" --training "./phrases_no_exclude_train.jsonl" --dev "./phrases_no_exclude_dev.jsonl" --test "./phrases_no_exclude_test.jsonl"
```

## Steps 
### 1. Text corpus indexing  
We consider two main sources from the text corpus, Wiki-PM (WikiMed and PubMedDS) and MIM-III(MIMIC-III discharge summaries). After splitting into text passages, we index each separately.
Example for indexing Wiki-PM (Wiki_pubMed_splitted):

```
python3 "preprocessing/elasticsearch_indexing.py" --data_path"./wikimed_pubmed_splitted.txt"
```

### 2. Retrieving the top-50 using the TSDAE model 
We then pass the pre-processed dataset (if needs pre-processing) along with the index name and the TSDAE model, to get the new CSV files with the top-50 passages, Here is an example for MedQA:
```
python3 "preprocessing/top50_by_tsdae.py" --train "./MedQA_training.csv" --dev "./MedQA_dev.csv" --testing "./MedQA_test.csv" --TSDAE_path "./TSDAE" --index_name "Wiki_pubMed_splitted"
```
### 3. Cross-Encoder pretraining 
We consider four BERT variants: BERT, ClinicalBERT, SciBERT, PubMedBERT to pretrain a cross encoder on one of the pretraining datasets.
We use the implementation from [training_stsbenchmark.py](https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/cross-encoder/training_stsbenchmark.py) for pretraining a cross encoder on, for example, STS-B dataset and save a checkpoint. We name the resulted model SciBERT_STS, for example, if initialized from SciBERT . 

### 4. Constructing the distantly supervised dataset and fine-tuning the final cross-encoder 

```
python3 "evaluate.py" --train "./MedQA_training_with_top50.csv" --dev "./MedQA_dev_with_top50.csv" --testing "./MedQA_test_with_top50.csv" --seed_val 12345  --pretrained_cross_encoder_path "./SciBERT_STS" --top_k 5 --temp_dir "./temp" --model_path "./scibert_scivocab_cased/"
```
___
## Citation
```
@inproceedings{10.1145/3477495.3532003,
author = {Alghanmi, Israa and Espinosa-Anke, Luis and Schockaert, Steven},
title = {Interpreting Patient Descriptions Using Distantly Supervised Similar Case Retrieval},
year = {2022},
isbn = {9781450387323},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3477495.3532003},
doi = {10.1145/3477495.3532003},
abstract = {Biomedical natural language processing often involves the interpretation of patient descriptions, for instance for diagnosis or for recommending treatments. Current methods, based on biomedical language models, have been found to struggle with such tasks. Moreover, retrieval augmented strategies have only had limited success, as it is rare to find sentences which express the exact type of knowledge that is needed for interpreting a given patient description. For this reason, rather than attempting to retrieve explicit medical knowledge, we instead propose to rely on a nearest neighbour strategy. First, we retrieve text passages that are similar to the given patient description, and are thus likely to describe patients in similar situations, while also mentioning some hypothesis (e.g. a possible diagnosis of the patient). We then judge the likelihood of the hypothesis based on the similarity of the retrieved passages. Identifying similar cases is challenging, however, as descriptions of similar patients may superficially look rather different, among others because they often contain an abundance of irrelevant details. To address this challenge, we propose a strategy that relies on a distantly supervised cross-encoder. Despite its conceptual simplicity, we find this strategy to be effective in practice.},
booktitle = {Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval},
pages = {460–470},
numpages = {11},
keywords = {distant supervision, biomedical nlp, similar case retrieval},
location = {Madrid, Spain},
series = {SIGIR '22}
}
```
