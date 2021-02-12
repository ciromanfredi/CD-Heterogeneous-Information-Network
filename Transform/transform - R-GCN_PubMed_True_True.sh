#!/bin/bash

# Note: Only 'R-GCN', 'HAN', 'MAGNN', and 'HGT' support attributed='True' or supervised='True'
# Note: Only 'DBLP' and 'PubMed' contain node attributes.

dataset='PubMed' # choose from 'DBLP', 'Yelp', 'Freebase', and 'PubMed'
model='R-GCN' # choose from 'metapath2vec-ESim', 'PTE', 'HIN2Vec', 'AspEm', 'HEER', 'R-GCN', 'HAN', 'MAGNN', 'HGT', 'TransE', 'DistMult', 'ComplEx', and 'ConvE'
attributed='True' # choose 'True' or 'False'
supervised='True' # choose 'True' or 'False'

mkdir ../Model/${model}/data
mkdir ../Model/${model}/data/${dataset}_${attributed}_${supervised}

python transform.py -dataset ${dataset} -model ${model} -attributed ${attributed} -supervised ${supervised}