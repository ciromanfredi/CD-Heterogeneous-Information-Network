## Evaluate

This stage evaluates the output embeddings based on specific tasks.

Users need to specify the following input parameters when calling the function 
```community_detection_personale(name_model,dataset,attributed,supervised)```:
- **model**: choose from ```HIN2Vec```,```R-GCN```,```HAN```,```ComplEx```
- **dataset**: choose from ```Yelp``` and ```PubMed```;
- **attributed**: choose ```True``` for attributed training or ```False``` for unattributed training;
- **supervised**: choose ```True``` for semi-supervised training or ```False``` for unsupervised training.

*Note: Only Message-Passing Methods (```R-GCN```, ```HAN```) support attributed or semi-supervised training.* <br /> 
*Note: Only ```PubMed``` contain node attributes.*

**Metric Calculation**: <br />
- **PubMed** : We train 7 Classifiers (SVC, Logistic Regression, Random Forest, MLP-Adam, SKRanger - Catboost - Xgboost) based on the learned embeddings on 80% of the labeled nodes and predict on the remaining 20%. We repeat the process for standard five-fold cross validation and compute the average scores regarding :
 ```Macro-F1```,```Micro-F1```,```Accuracy```,```Precision```,```Recall```

- **Yelp** : We train 4 Classifiers (SVC,Logistic Regression, Random Forest, MLP-Adam) to perform binary classification in order to address the multi-label classification problem. The training is based on the learned embeddings on 80% of the labeled nodes and the prediction is on the remaining 20%. We repeat the process for standard five-fold cross validation and compute the average scores regarding:
 ```Macro-F1```,```Micro-F1```,```Accuracy```,```Precision```,```Recall```

Run :
-**```from CDHIN import community_detection_personale```**
-**```community_detection_personale(name_model,dataset,attributed,supervised)```**
to complete *Stage 4: Evaluate*.

The evaluation results are stored in :
-**```PubMedScore.txt```** if you choose PubMed dataset,
-**```YelpScore.txt```** if you choose Yelp dataset