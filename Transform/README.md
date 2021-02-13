## Transform

This stage transforms a dataset from its original format to the training input format.

Users need to specify the following parameters in ```transform.sh```:
- **dataset**: choose from ```Yelp```and ```PubMed```;
- **model**: choose from ```HIN2Vec``````R-GCN```, ```HAN```,```ComplEx```;
- **attributed**: choose ```True``` for attributed training or ```False``` for unattributed training;
- **supervised**: choose ```True``` for semi-supervised training or ```False``` for unsupervised training.

*Note: Only Message-Passing Methods (```R-GCN```, ```HAN```) support attributed or semi-supervised training.* <br /> 
*Note: Only ```PubMed``` contain node attributes.*

Run ```bash transform - model_dataset_attributed_supervised.sh``` to complete *Stage 2: Transform*.