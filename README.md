# Heterogeneous Network Representation Learning: Benchmark with Data and Code

## Citation

Please cite the following work if you find the data/code useful.

```
@article{yang2020heterogeneous,
  title={Heterogeneous Network Representation Learning: A Unified Framework with Survey and Benchmark},
  author={Yang, Carl and Xiao, Yuxin and Zhang, Yu and Sun, Yizhou and Han, Jiawei},
  journal={TKDE},
  year={2020}
}
```

## Guideline

### Stage 1: Data

We provide 2 HIN benchmark datasets: ```Yelp``` and ```PubMed```.

Each dataset contains:
- 3 data files (```node.dat```, ```link.dat```, ```label.dat```);
- 2 evaluation files (```link.dat.test```, ```label.dat.test```);
- 2 description files (```meta.dat```, ```info.dat```);
- 1 recording file (```record.dat```).

Please refer to the ```Data``` folder for more details.

### Stage 2: Transform

This stage transforms a dataset from its original format to the training input format.

Users need to specify the targeting dataset, the targeting model, and the training settings.

Please refer to the ```Transform``` folder for more details.

### Stage 3: Model

We provide 4 HIN baseline implementaions: 
- 1 Proximity-Preserving Methods (```HIN2Vec```, ```AspEm```); 
- 2 Message-Passing Methods (```R-GCN```, ```HAN```); 
- 1 Relation-Learning Methods (```ComplEx```).

Please refer to the ```Model``` folder for more details.

### Stage 4: Evaluate

This stage evaluates the output embeddings based on specific tasks. 

Users need to specify the targeting dataset, the targeting model, and the evaluation tasks.

Please refer to the ```Evaluate``` folder for more details.
