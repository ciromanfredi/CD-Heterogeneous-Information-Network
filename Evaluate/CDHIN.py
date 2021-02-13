import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.cluster import KMeans
import networkx as nx
import networkx.algorithms.community as nxcom
float_formatter = "{:.3f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})
    
def creoTrainTestLabel(label_file_path,label_test_path,emb_dict):
    train_index, train_labels, train_embeddings =[], [], []
    with open(label_file_path,'r') as label_file:
        for line in label_file:
            index, _, _, label = line[:-1].split('\t')
            train_index.append(index)
            train_labels.append(label)
            train_embeddings.append(emb_dict[index])    
    train_index,train_labels, train_embeddings =np.array(train_index).astype(int), np.array(train_labels).astype(int), np.array(train_embeddings)  
    
    test_index, test_labels, test_embeddings =[], [], []
    with open(label_test_path,'r') as label_file:
        for line in label_file:
            index, _, _, label = line[:-1].split('\t')
            test_index.append(index)
            test_labels.append(label)
            test_embeddings.append(emb_dict[index])    
    test_index, test_labels, test_embeddings =np.array(test_index).astype(int), np.array(test_labels).astype(int), np.array(test_embeddings)
    return train_index,test_index,train_labels,train_embeddings,test_labels,test_embeddings

def load(emb_file_path):
  emb_dict = {}
  with open(emb_file_path,'r') as emb_file:        
      for i, line in enumerate(emb_file):
          if i == 0:
              train_para = line[:-1]
          else:
              index, emb = line[:-1].split('\t')
              emb_dict[index] = np.array(emb.split()).astype(np.float32)
  return train_para, emb_dict

def TSNEImpl(embeddings,perplexity=None):
  print('STARTING TSNE')
  if perplexity:
      tsne_model = TSNE(n_components=2, perplexity=perplexity, init='pca', n_iter=2500, random_state=23)
  else:
      tsne_model = TSNE(n_components=2, init='pca', n_iter=2500, random_state=23)
  X=tsne_model.fit_transform(embeddings)
  return X

def PLOTImpl(X,Y,name,dim=0.7,axis=None):
  print('STARTING PLOT')
  if axis :
    plt.axis(axis)
  plt.figure(figsize=(20,20))
  scatter = plt.scatter(X[:, 0], X[:, 1], c=Y, label=Y, s=dim)
  plt.legend(*scatter.legend_elements(num=len(set(Y))), title="Classes", loc="upper center", ncol=len(set(Y)))
  plt.savefig(name)
  plt.clf()

def CreoEmbFromDictEmb(emb_dict):
  print('STARTING EMBEDDING CREATION (from dict)')
  indexlist, embeddings =[], []
  for key,value in emb_dict.items():
      indexlist.append(key)
      embeddings.append(value)
  indexlist,embeddings=np.array(indexlist).astype(int),np.array(embeddings).astype(np.float32)
  return indexlist,embeddings

def CreoEmbFromPath(emb_file_path):
    print('STARTING EMBEDDING CREATION (from path)')
    embeddings=[]
    with open(emb_file_path,'r') as emb_file:        
        for i, line in enumerate(emb_file):
            if i == 0:
                train_para = line[:-1]
            else:
                index, emb = line[:-1].split('\t')
                embeddings.append([float(x) for x in emb.split()])
        embeddings=np.array(embeddings).astype(np.float32)
    return train_para, embeddings

def writeScore(name_model,dataset,attributed,supervised,macro,micro,accuracy,precision,recall,classifier):
    f = open("/content/drive/MyDrive/Colab Notebooks/HNE-master/Evaluate/"+dataset+"_Score.txt", "a+")
    f.write("Model: "+str(name_model)+", Dataset: "+str(dataset)+", Attributed: "+str(attributed)+", Supervised: "+str(supervised)+", Classifier: "+str(classifier))
    f.write('\nMacro-F1: '+str(np.mean(macro))+' STD: '+str(np.std(macro)))
    f.write('\nMicro-F1: '+str(np.mean(micro))+' STD: '+str(np.std(micro)))
    f.write('\nAccuracy: '+str(np.mean(accuracy))+' STD: '+str(np.std(accuracy)))
    f.write('\nPrecision: '+str(np.mean(precision))+' STD: '+str(np.std(precision)))
    f.write('\nRecall: '+str(np.mean(recall))+' STD: '+str(np.std(recall)))
    f.write("\n\n")
    f.close()

def writeClusteringScore(name_model,dataset,attributed,supervised,nmi_list,ari_list):
    f = open("/content/drive/MyDrive/Colab Notebooks/HNE-master/Evaluate/"+dataset+"_Score.txt", "a+")
    f.write("Model: "+str(name_model)+", Dataset: "+str(dataset)+", Attributed: "+str(attributed)+", Supervised: "+str(supervised)+"\n")
    f.write('NMI_score: ')
    f.write(str(np.mean(nmi_list))+ ", StdNMI: " + str(np.std(nmi_list)))
    f.write(', ARI_score: ')
    f.write(str(np.mean(ari_list)) +", StdARI: "+ str(np.std(ari_list)))
    f.write("\n\n")
    f.close()

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

def single_label_metrics(test_labels,preds):
    macro=f1_score(test_labels, preds, average='macro')
    micro=f1_score(test_labels, preds, average='micro')
    accuracy=accuracy_score(test_labels,preds)
    precision=precision_score(test_labels, preds, average="macro")
    recall=recall_score(test_labels, preds, average="macro")
    return preds,macro,micro,accuracy,precision,recall

def multi_label_metrics(test_labels,preds):
    scores=f1_score(test_labels, preds, average='binary')
    accuracylist=accuracy_score(test_labels,preds)
    precisionlist=precision_score(test_labels, preds, average="binary")
    recalllist=recall_score(test_labels, preds, average="binary")
    return scores, accuracylist, precisionlist, recalllist

from sklearn.svm import LinearSVC
def SVCImpl(train_embeddings, train_labels,test_embeddings,test_labels,seed,max_iter,single_label=True):
    clf = LinearSVC(random_state=seed, max_iter=max_iter)
    clf.fit(train_embeddings, train_labels)
    predssvc = clf.predict(test_embeddings)
    if single_label==True:
        return single_label_metrics(test_labels,predssvc)
    else:
        return multi_label_metrics(test_labels,predssvc)

from sklearn.linear_model import LogisticRegression
def LogRegImpl(train_embeddings, train_labels,test_embeddings,test_labels,seed,max_iter,single_label=True): 
    logreg = LogisticRegression(random_state=seed, max_iter=max_iter)
    logreg.fit(train_embeddings, train_labels)
    predslogreg = logreg.predict(test_embeddings)
    if single_label==True:
        return single_label_metrics(test_labels,predslogreg)
    else:
        return multi_label_metrics(test_labels,predslogreg)

from sklearn.ensemble import RandomForestClassifier
def RFImpl(train_embeddings, train_labels,test_embeddings,test_labels,seed,max_iter,single_label=True): 
    rf = RandomForestClassifier()
    rf.fit(train_embeddings, train_labels)
    predsrf = rf.predict(test_embeddings)
    if single_label==True:
        return single_label_metrics(test_labels,predsrf)
    else:
        return multi_label_metrics(test_labels,predsrf)

from sklearn.neural_network import MLPClassifier
def ADAMImpl(train_embeddings, train_labels,test_embeddings,test_labels,seed,max_iter,single_label=True):
    
    adam=MLPClassifier(max_iter=max_iter)
    adam.fit(train_embeddings, train_labels)
    predsadam = adam.predict(test_embeddings)
    if single_label==True:
        return single_label_metrics(test_labels,predsadam)
    else:
        return multi_label_metrics(test_labels,predsadam)

from skranger.ensemble import RangerForestClassifier
def RangerForestIMPL(train_embeddings, train_labels,test_embeddings,test_labels,single_label=True):
    rfc = RangerForestClassifier()
    rfc.fit(train_embeddings, train_labels)
    predsranger = rfc.predict(test_embeddings)
    if single_label==True:
        return single_label_metrics(test_labels,predsranger)

from catboost import CatBoostClassifier
def CatboostIMPL(train_embeddings, train_labels,test_embeddings,test_labels,single_label=True,iterations=2,learning_rate=1,depth=2):
    catboost = CatBoostClassifier(iterations=iterations,learning_rate=learning_rate,depth=depth)
    catboost.fit(train_embeddings, train_labels)
    predscatboost = catboost.predict(test_embeddings)
    if single_label==True:
        return single_label_metrics(test_labels,predscatboost)

import xgboost as xgb
def xgboostIMPL(train_embeddings, train_labels,test_embeddings,test_labels,single_label=True,iterations=2,learning_rate=1,depth=2):
    xgb_model = xgb.XGBClassifier(n_jobs=1).fit(train_embeddings, train_labels)
    predsxgb = xgb_model.predict(test_embeddings)
    if single_label==True:
        return single_label_metrics(test_labels,predsxgb)

from sklearn.metrics import confusion_matrix
def ConfusionMatrixIMPL(test_labels,preds,txt="Confusion Matrix"):
    cm = confusion_matrix(test_labels,preds)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(txt)
    print(cm)   

def unsupervised_single_class_single_label(full_embeddings, full_labels,n_splits=5,seed=21,max_iter=10000):    
    print("Start: unsupervised_single_class_single_label")
    macrolistsvc,microlistsvc,accuracylistsvc,precisionlistsvc,recalllistsvc=[],[],[],[],[]
    macrolistlogreg,microlistlogreg,accuracylistlogreg,precisionlistlogreg,recalllistlogreg=[],[],[],[],[]
    macrolistrf,microlistrf,accuracylistrf,precisionlistrf,recalllistrf=[],[],[],[],[]
    macrolistadam,microlistadam,accuracylistadam,precisionlistadam,recalllistadam=[],[],[],[],[]
    macrolistranger,microlistranger,accuracylistranger,precisionlistranger,recalllistranger=[],[],[],[],[]
    macrolistcatboost,microlistcatboost,accuracylistcatboost,precisionlistcatboost,recalllistcatboost=[],[],[],[],[]
    macrolistxgboost,microlistxgboost,accuracylistxgboost,precisionlistxgboost,recalllistxgboost=[],[],[],[],[]
    
    skf = StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=seed)
    index=0
    for train_idx, test_idx in skf.split(full_embeddings, full_labels):

        predssvc, macrosvc, microsvc, accuracysvc, precisionsvc, recallsvc=SVCImpl(full_embeddings[train_idx], full_labels[train_idx],full_embeddings[test_idx],full_labels[test_idx],seed,max_iter)
        predslogreg,macrologreg,micrologreg,accuracylogreg,precisionlogreg,recalllogreg=LogRegImpl(full_embeddings[train_idx], full_labels[train_idx],full_embeddings[test_idx],full_labels[test_idx],seed,max_iter)
        predsrf,macrorf,microrf,accuracyrf,precisionrf,recallrf=RFImpl(full_embeddings[train_idx], full_labels[train_idx],full_embeddings[test_idx],full_labels[test_idx],seed,max_iter)
        predsadam,macroadam,microadam,accuracyadam,precisionadam,recalladam=ADAMImpl(full_embeddings[train_idx], full_labels[train_idx],full_embeddings[test_idx],full_labels[test_idx],seed,max_iter)
        predsranger,macroranger,microranger,accuracyranger,precisionranger,recallranger=RangerForestIMPL(full_embeddings[train_idx], full_labels[train_idx],full_embeddings[test_idx],full_labels[test_idx])
        predscatboost,macrocatboost,microcatboost,accuracycatboost,precisioncatboost,recallcatboost=CatboostIMPL(full_embeddings[train_idx], full_labels[train_idx],full_embeddings[test_idx],full_labels[test_idx])
        predsxgboost,macroxgboost,microxgboost,accuracyxgboost,precisionxgboost,recallxgboost=xgboostIMPL(full_embeddings[train_idx], full_labels[train_idx],full_embeddings[test_idx],full_labels[test_idx])
        
        macrolistsvc.append(macrosvc)
        microlistsvc.append(microsvc)
        accuracylistsvc.append(accuracysvc)
        precisionlistsvc.append(precisionsvc)
        recalllistsvc.append(recallsvc)

        macrolistlogreg.append(macrologreg)
        microlistlogreg.append(micrologreg)
        accuracylistlogreg.append(accuracylogreg)
        precisionlistlogreg.append(precisionlogreg)
        recalllistlogreg.append(recalllogreg)

        macrolistrf.append(macrorf)
        microlistrf.append(microrf)
        accuracylistrf.append(accuracyrf)
        precisionlistrf.append(precisionrf)
        recalllistrf.append(recallrf)

        macrolistadam.append(macroadam)
        microlistadam.append(microadam)
        accuracylistadam.append(accuracyadam)
        precisionlistadam.append(precisionadam)
        recalllistadam.append(recalladam)

        macrolistranger.append(macroranger)
        microlistranger.append(microranger)
        accuracylistranger.append(accuracyranger)
        precisionlistranger.append(precisionranger)
        recalllistranger.append(recallranger)

        macrolistcatboost.append(macrocatboost)
        microlistcatboost.append(microcatboost)
        accuracylistcatboost.append(accuracycatboost)
        precisionlistcatboost.append(precisioncatboost)
        recalllistcatboost.append(recallcatboost)

        macrolistxgboost.append(macroxgboost)
        microlistxgboost.append(microxgboost)
        accuracylistxgboost.append(accuracyxgboost)
        precisionlistxgboost.append(precisionxgboost)
        recalllistxgboost.append(recallxgboost)

        if index==0:
            ConfusionMatrixIMPL(full_labels[test_idx],predssvc,"Confusion Matrix Support Vector Classifier")
            ConfusionMatrixIMPL(full_labels[test_idx],predslogreg,"Confusion Matrix Logistic Regression")
            ConfusionMatrixIMPL(full_labels[test_idx],predsrf,"Confusion Matrix Random Forest")
            ConfusionMatrixIMPL(full_labels[test_idx],predsadam,"Confusion Matrix MLP (ADAM optmizer)")            
            ConfusionMatrixIMPL(full_labels[test_idx],predsranger,"Confusion Matrix Ranger Random Forest")
            ConfusionMatrixIMPL(full_labels[test_idx],predscatboost,"Confusion Matrix Catboost")
            ConfusionMatrixIMPL(full_labels[test_idx],predsxgboost,"Confusion Matrix Xgboost")
        index=index+1

    writeScore(name_model,dataset,attributed,supervised,macrolistsvc,microlistsvc,accuracylistsvc,precisionlistsvc,recalllistsvc,"SVC")   
    writeScore(name_model,dataset,attributed,supervised,macrolistlogreg,microlistlogreg,accuracylistlogreg,precisionlistlogreg,recalllistlogreg,"Logistic Regression")   
    writeScore(name_model,dataset,attributed,supervised,macrolistrf,microlistrf,accuracylistrf,precisionlistrf,recalllistrf,"Random Forest")   
    writeScore(name_model,dataset,attributed,supervised,macrolistadam,microlistadam,accuracylistadam,precisionlistadam,recalllistadam,"MLP-Adam") 
    writeScore(name_model,dataset,attributed,supervised,macrolistranger,microlistranger,accuracylistranger,precisionlistranger,recalllistranger,"Ranger Random Forest")   
    writeScore(name_model,dataset,attributed,supervised,macrolistcatboost,microlistcatboost,accuracylistcatboost,precisionlistcatboost,recalllistcatboost,"Catboost")   
    writeScore(name_model,dataset,attributed,supervised,macrolistxgboost,microlistxgboost,accuracylistxgboost,precisionlistxgboost,recalllistxgboost,"Xgboost") 
    #return np.mean(micro), np.mean(macro), np.mean(accuracy), np.mean(precision), np.mean(recall)

def semisupervised_single_class_single_label(train_embeddings, train_labels, test_embeddings,test_labels,seed=21, max_iter=100000):  
    print("Start: semisupervised_single_class_single_label")
       
    predssvc, macrosvc, microsvc, accuracysvc, precisionsvc, recallsvc=SVCImpl(train_embeddings, train_labels,test_embeddings,test_labels,seed,max_iter)
    predslogreg,macrologreg,micrologreg,accuracylogreg,precisionlogreg,recalllogreg=LogRegImpl(train_embeddings, train_labels,test_embeddings,test_labels,seed,max_iter)
    predsrf,macrorf,microrf,accuracyrf,precisionrf,recallrf=RFImpl(train_embeddings, train_labels,test_embeddings,test_labels,seed,max_iter)
    predsadam,macroadam,microadam,accuracyadam,precisionadam,recalladam=ADAMImpl(train_embeddings, train_labels,test_embeddings,test_labels,seed,max_iter)
    predsranger,macroranger,microranger,accuracyranger,precisionranger,recallranger=RangerForestIMPL(train_embeddings, train_labels,test_embeddings,test_labels)
    predscatboost,macrocatboost,microcatboost,accuracycatboost,precisioncatboost,recallcatboost=CatboostIMPL(train_embeddings, train_labels,test_embeddings,test_labels)
    predsxgboost,macroxgboost,microxgboost,accuracyxgboost,precisionxgboost,recallxgboost=xgboostIMPL(train_embeddings, train_labels,test_embeddings,test_labels)
    
    ConfusionMatrixIMPL(test_labels,predssvc,"Confusion Matrix Support Vector Classifier")
    ConfusionMatrixIMPL(test_labels,predslogreg,"Confusion Matrix Logistic Regression")
    ConfusionMatrixIMPL(test_labels,predsrf,"Confusion Matrix Random Forest")
    ConfusionMatrixIMPL(test_labels,predsadam,"Confusion Matrix MLP (ADAM optmizer)")
    ConfusionMatrixIMPL(test_labels,predsranger,"Confusion Matrix Ranger Random Forest")
    ConfusionMatrixIMPL(test_labels,predscatboost,"Confusion Matrix Catboost")
    ConfusionMatrixIMPL(test_labels,predsxgboost,"Confusion Matrix Xgboost")

    writeScore(name_model,dataset,attributed,supervised,macrosvc,microsvc,accuracysvc,precisionsvc,recallsvc,"SVC")   
    writeScore(name_model,dataset,attributed,supervised,macrologreg,micrologreg,accuracylogreg,precisionlogreg,recalllogreg,"Logistic Regression")   
    writeScore(name_model,dataset,attributed,supervised,macrorf,microrf,accuracyrf,precisionrf,recallrf,"Random Forest")   
    writeScore(name_model,dataset,attributed,supervised,macroadam,microadam,accuracyadam,precisionadam,recalladam,"MLP-Adam") 
    writeScore(name_model,dataset,attributed,supervised,macroranger,microranger,accuracyranger,precisionranger,recallranger,"Ranger Random Forest")   
    writeScore(name_model,dataset,attributed,supervised,macrocatboost,microcatboost,accuracycatboost,precisioncatboost,recallcatboost,"Catboost")   
    writeScore(name_model,dataset,attributed,supervised,macroxgboost,microxgboost,accuracyxgboost,precisionxgboost,recallxgboost,"Xgboost")   
    #return np.mean(macro), np.mean(macro), np.mean(accuracy), np.mean(precision), np.mean(recall)

def unsupervised_single_class_multi_label(label_file_path, label_test_path, emb_dict, seed=21, max_iter=10000):
    nodes_count, binary_labels, label_dict, label_count, labeled_nodes = len(emb_dict), [], {}, 0, set()
    for file_path in [label_file_path, label_test_path]:
        with open(file_path,'r') as label_file:
            for line in label_file:
                index, _, nclass, label = line[:-1].split('\t')   
                for each in label.split(','):                
                    if (nclass, each) not in label_dict:
                        label_dict[(nclass, each)] = label_count
                        label_count += 1
                        binary_labels.append(np.zeros(nodes_count).astype(np.bool_))
                    binary_labels[label_dict[(nclass, each)]][int(index)] = True
                    labeled_nodes.add(int(index))
    labeled_nodes = np.sort(list(labeled_nodes))
    binary_labels = np.array(binary_labels)[:,labeled_nodes]
    
    embs = []
    for index in labeled_nodes:
        embs.append(emb_dict[str(index)])
    embs = np.array(embs)
    
    weights=[]
    total_scoressvc,accuracylist2svc,precisionlist2svc,recalllist2svc = [], [], [], []
    total_scoreslogreg,accuracylist2logreg,precisionlist2logreg,recalllist2logreg = [], [], [], []
    total_scoresrf,accuracylist2rf,precisionlist2rf,recalllist2rf = [], [], [], []
    total_scoresadam,accuracylist2adam,precisionlist2adam,recalllist2adam = [], [], [], []
    
    for ntype, binary_label in enumerate(binary_labels):
        
        scoressvc,accuracylistsvc,precisionlistsvc,recalllistsvc = [], [], [], []
        scoreslogreg,accuracylistlogreg,precisionlistlogreg,recalllistlogreg = [], [], [], []
        scoresrf,accuracylistrf,precisionlistrf,recalllistrf = [], [], [], []
        scoresadam,accuracylistadam,precisionlistadam,recalllistadam = [], [], [], []

        skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=seed)
        for train_idx, test_idx in skf.split(embs, binary_label):            

            scoresvc,accuracysvc,precisionsvc,recallsvc=SVCImpl(embs[train_idx], binary_label[train_idx],embs[test_idx],binary_label[test_idx],seed,max_iter,single_label=False)
            scorelogreg,accuracylogreg,precisionlogreg,recalllogreg=LogRegImpl(embs[train_idx], binary_label[train_idx],embs[test_idx],binary_label[test_idx],seed,max_iter,single_label=False)
            scorerf,accuracyrf,precisionrf,recallrf=RFImpl(embs[train_idx], binary_label[train_idx],embs[test_idx],binary_label[test_idx],seed,max_iter,single_label=False)
            scoreadam,accuracyadam,precisionadam,recalladam=ADAMImpl(embs[train_idx], binary_label[train_idx],embs[test_idx],binary_label[test_idx],seed,max_iter,single_label=False)

            scoressvc.append(scoresvc)
            accuracylistsvc.append(accuracysvc)
            precisionlistsvc.append(precisionsvc)
            recalllistsvc.append(recallsvc)

            scoreslogreg.append(scorelogreg)
            accuracylistlogreg.append(accuracylogreg)
            precisionlistlogreg.append(precisionlogreg)
            recalllistlogreg.append(recalllogreg)

            scoresrf.append(scorerf)
            accuracylistrf.append(accuracyrf)
            precisionlistrf.append(precisionrf)
            recalllistrf.append(recallrf)

            scoresadam.append(scoreadam)
            accuracylistadam.append(accuracyadam)
            precisionlistadam.append(precisionadam)
            recalllistadam.append(recalladam)

        weights.append(sum(binary_label))

        total_scoressvc.append(np.mean(scoressvc))
        accuracylist2svc.append(np.mean(accuracylistsvc))
        precisionlist2svc.append(np.mean(precisionlistsvc))
        recalllist2svc.append(np.mean(recalllistsvc))

        total_scoreslogreg.append(np.mean(scoreslogreg))
        accuracylist2logreg.append(np.mean(accuracylistlogreg))
        precisionlist2logreg.append(np.mean(precisionlistlogreg))
        recalllist2logreg.append(np.mean(recalllistlogreg))

        total_scoresrf.append(np.mean(scoresrf))
        accuracylist2rf.append(np.mean(accuracylistrf))
        precisionlist2rf.append(np.mean(precisionlistrf))
        recalllist2rf.append(np.mean(recalllistrf))

        total_scoresadam.append(np.mean(scoresadam))
        accuracylist2adam.append(np.mean(accuracylistadam))
        precisionlist2adam.append(np.mean(precisionlistadam))
        recalllist2adam.append(np.mean(recalllistadam))

    macrosvc = sum(total_scoressvc)/len(total_scoressvc)
    microsvc = sum([score*weight for score, weight in zip(total_scoressvc, weights)])/sum(weights)

    macrologreg = sum(total_scoreslogreg)/len(total_scoreslogreg)
    micrologreg = sum([score*weight for score, weight in zip(total_scoreslogreg, weights)])/sum(weights)

    macrorf = sum(total_scoresrf)/len(total_scoresrf)
    microrf = sum([score*weight for score, weight in zip(total_scoresrf, weights)])/sum(weights)

    macrorf = sum(total_scoresrf)/len(total_scoresrf)
    microadam = sum([score*weight for score, weight in zip(total_scoresadam, weights)])/sum(weights)
    
    writeScore(name_model,dataset,attributed,supervised,macrosvc,microsvc,np.mean(accuracylist2svc),np.mean(precisionlist2svc),np.mean(recalllist2svc),"SVC")    
    writeScore(name_model,dataset,attributed,supervised,macrologreg,micrologreg,np.mean(accuracylist2logreg),np.mean(precisionlist2logreg),np.mean(recalllist2logreg),"Logistic Regression")    
    writeScore(name_model,dataset,attributed,supervised,macrorf,microrf,np.mean(accuracylist2rf),np.mean(precisionlist2rf),np.mean(recalllist2rf),"Random Forest")    
    writeScore(name_model,dataset,attributed,supervised,macrorf,microadam,np.mean(accuracylist2adam),np.mean(precisionlist2adam),np.mean(recalllist2adam),"MLP-Adam")    
    #return macro, micro, accuracymean, precisionmean, recallmean

def semisupervised_single_class_multi_label(label_file_path, label_test_path, emb_dict, seed=21, max_iter=10000):

    nodes_count, binary_labels, label_dict, label_count, train_nodes, test_nodes = len(emb_dict), [], {}, 0, set(), set()
    for file_path in [label_file_path, label_test_path]:
        with open(file_path,'r') as label_file:
            for line in label_file:
                index, _, nclass, label = line[:-1].split('\t')   
                for each in label.split(','):                
                    if (nclass, each) not in label_dict:
                        label_dict[(nclass, each)] = label_count
                        label_count += 1
                        binary_labels.append(np.zeros(nodes_count).astype(np.bool_))
                    binary_labels[label_dict[(nclass, each)]][int(index)] = True
                    if file_path==label_file_path: train_nodes.add(int(index))
                    else: test_nodes.add(int(index))
    train_nodes, test_nodes = np.sort(list(train_nodes)), np.sort(list(test_nodes))
    train_labels, test_labels = np.array(binary_labels)[:,train_nodes], np.array(binary_labels)[:,test_nodes]
    
    train_embs, test_embs = [], []
    for index in train_nodes:
        train_embs.append(emb_dict[str(index)])
    for index in test_nodes:
        test_embs.append(emb_dict[str(index)])
    train_embs, test_embs = np.array(train_embs), np.array(test_embs)
    
    weights=[]
    accuracylistsvc,precisionlistsvc,recalllistsvc,total_scoressvc = [], [], [], []
    accuracylistlogreg,precisionlistlogreg,recalllistlogreg, total_scoreslogreg = [], [], [], []
    accuracylistrf,precisionlistrf,recalllistrf, total_scoresrf = [], [], [], []
    accuracylistadam,precisionlistadam,recalllistadam, total_scoresadam=[],[],[],[]

    for ntype, (train_label, test_label) in enumerate(zip(train_labels, test_labels)):           

        scoressvc,accuracysvc,precisionsvc,recallsvc=SVCImpl(train_embs, train_label,test_embs,test_label,seed,max_iter,single_label=False)
        scoreslogreg,accuracylogreg,precisionlogreg,recalllogreg=LogRegImpl(train_embs, train_label,test_embs,test_label,seed,max_iter,single_label=False)
        scoresrf,accuracyrf,precisionrf,recallrf=RFImpl(train_embs, train_label,test_embs,test_label,seed,max_iter,single_label=False)
        scoresadam,accuracyadam,precisionadam,recalladam=ADAMImpl(train_embs, train_label,test_embs,test_label,seed,max_iter,single_label=False)

        weights.append(sum(test_label))
        
        total_scoressvc.append(scoressvc)
        total_scoreslogreg.append(scoreslogreg)
        total_scoresrf.append(scoresrf)
        total_scoresadam.append(scoresadam)

        accuracylistsvc.append(accuracysvc)
        accuracylistlogreg.append(accuracylogreg)
        accuracylistrf.append(accuracyrf)
        accuracylistadam.append(accuracyadam)

        precisionlistsvc.append(precisionsvc)
        precisionlistlogreg.append(precisionlogreg)
        precisionlistrf.append(precisionrf)
        precisionlistadam.append(precisionadam)

        recalllistsvc.append(recallsvc)
        recalllistlogreg.append(recalllogreg)
        recalllistrf.append(recallrf)
        recalllistadam.append(recalladam)
        
    macrosvc = sum(total_scoressvc)/len(total_scoressvc)
    microsvc = sum([score*weight for score, weight in zip(total_scoressvc, weights)])/sum(weights)

    macrologreg = sum(total_scoreslogreg)/len(total_scoreslogreg)
    micrologreg = sum([score*weight for score, weight in zip(total_scoreslogreg, weights)])/sum(weights)

    macrorf = sum(total_scoresrf)/len(total_scoresrf)
    microrf = sum([score*weight for score, weight in zip(total_scoresrf, weights)])/sum(weights)

    macrorf = sum(total_scoresadam)/len(total_scoresadam)
    microadam = sum([score*weight for score, weight in zip(total_scoresadam, weights)])/sum(weights)

    writeScore(name_model,dataset,attributed,supervised,macrosvc,microsvc,np.mean(accuracylistsvc),np.mean(precisionlistsvc),np.mean(recalllistsvc),"SVC")    
    writeScore(name_model,dataset,attributed,supervised,macrologreg,micrologreg,np.mean(accuracylistlogreg),np.mean(precisionlistlogreg),np.mean(recalllistlogreg),"Logistic Regression")    
    writeScore(name_model,dataset,attributed,supervised,macrorf,microrf,np.mean(accuracylistrf),np.mean(precisionlistrf),np.mean(recalllistrf),"Random Forest")   
    writeScore(name_model,dataset,attributed,supervised,macrorf,microadam,np.mean(accuracylistadam),np.mean(precisionlistadam),np.mean(recalllistadam),"MLP-Adam") 
    #return macro, micro, accuracymean, precisionmean, recallmean

from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
def KMEANSImpl(train_embeddings, train_labels,testare_embeddings,n_clusters,label_originali=[],num_rip=None):
    print('STARTING KMEANS')
    if len(label_originali) > 0:
        print('WITH EVALUATION')
        nmi_list, ari_list = [], []

        for _ in range(num_rip):
            y_pred = KMEANSImpl(train_embeddings, train_labels,testare_embeddings,n_clusters)
            nmi_score = normalized_mutual_info_score(label_originali, y_pred, average_method='arithmetic')
            ari_score = adjusted_rand_score(label_originali, y_pred)
            nmi_list.append(nmi_score)
            ari_list.append(ari_score)

        writeClusteringScore(name_model,dataset,attributed,supervised,nmi_list,ari_list)
        return y_pred
    else:
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit_predict(train_embeddings, train_labels)
        labels_pred = kmeans.predict(testare_embeddings)
        return labels_pred

def creoGraphNetworkX(emb_dict):
    G = nx.Graph()

    #ADD NODES
    for index,embeddings in emb_dict.items():
        G.add_node(index, attr = list(embeddings))

    # ADD EDGE
    archi = open("/content/drive/MyDrive/Colab Notebooks/HNE-master/Data/"+dataset+"/link.dat")
    for arco in archi:
        elementi=str(arco).split('\t')
        if elementi[0] in G.nodes():
            G.add_edge(elementi[0],elementi[1],type=elementi[2],weight=int(elementi[3]))

    print('G.number_of_nodes()')
    print(G.number_of_nodes())
    print('G.number_of_edges()')
    print(G.number_of_edges())
    
    return G

# indexNodes: ['3333','4444','5555']
# labels :    ['9','10','9']
def fromIndexLabelCreateCommunityNetworkX(indexNodes,labels):
  index=0
  dizionario={}
  for label in set(labels):
    dizionario[label]=[]
  for label in labels:
    dizionario[label].append(str(indexNodes[index]))
    index=index+1

  listafrozenset=[]
  for key,value in dizionario.items():
    tupla=tuple(value)
    frozen=frozenset(tupla)
    listafrozenset.append(frozen)
  return listafrozenset

name_model=''
dataset=''
attributed=''
supervised=''
def community_detection_personale(name_model1,dataset1,attributed1,supervised1):
    global name_model
    name_model=name_model1
    global dataset
    dataset=dataset1
    global attributed
    attributed=attributed1
    global supervised
    supervised=supervised1

    label_file_path="/content/drive/MyDrive/Colab Notebooks/HNE-master/Data/"+dataset+"/label.dat"
    label_test_path="/content/drive/MyDrive/Colab Notebooks/HNE-master/Data/"+dataset+"/label.dat.test"


    stringadataset=''
    if attributed:
        stringadataset=stringadataset+'_True_'
    else:
        stringadataset=stringadataset+'_False_'
    if supervised:
        stringadataset=stringadataset+'True'
    else:
        stringadataset=stringadataset+'False'
    name_file=name_model+'_'+dataset+stringadataset

    emb_file_path="/content/drive/MyDrive/Colab Notebooks/HNE-master/Model/"+name_model+"/data/"+dataset+stringadataset+"/emb.dat"
    
    train_para, emb_dict=load(emb_file_path)

    print('Calcolo Micro, Macro, accuracy, precision, recall')
    if dataset=="PubMed":
        train_index,test_index,train_labels,train_embeddings,test_labels,test_embeddings=creoTrainTestLabel(label_file_path,label_test_path,emb_dict)
        full_embeddings = np.append(train_embeddings,test_embeddings,axis=0)
        full_labels = np.append(train_labels,test_labels,axis=0)
        if not supervised:
            unsupervised_single_class_single_label(full_embeddings, full_labels)
        else:
            semisupervised_single_class_single_label(train_embeddings, train_labels, test_embeddings,test_labels)
            #semisupervised_single_class_single_label_con_percentuale(train_embeddings, train_labels, test_embeddings,test_labels,num_rip=10)
    elif dataset=="Yelp":
        #train_labels,train_embeddings,test_labels,test_embeddings=transformYelpSingleLabel(label_file_path,label_test_path,emb_dict)
        #full_embeddings = np.append(train_embeddings,test_embeddings,axis=0)
        #full_labels = np.append(train_labels,test_labels,axis=0)
        if not supervised:
            unsupervised_single_class_multi_label(label_file_path, label_test_path, emb_dict)
        else:
            semisupervised_single_class_multi_label(label_file_path, label_test_path, emb_dict)
    
    n_clusters = len(set(full_labels))
      
    ######
    print('Caso 1')
    #Caso1 -> Kmeans addestrato su train_embeddings - train_labels, testato su test_embeddings
    #         Tsne senza perplexity 
    #         Plot dello spazio embeddings di test con le label predette dal kMeans
    test_labels_predette=KMEANSImpl(train_embeddings, train_labels,test_embeddings,n_clusters)
    test_embeddings_2D=TSNEImpl(test_embeddings)
    PLOTImpl(test_embeddings_2D,test_labels_predette,name_file+'Caso1.png',dim=40)

    #######
    print('Caso 2')
    #Caso3 -> Plot di tutto lo spazio embeddings (train + test embeddings) con label di default (prese dai file)     
    full_embeddings_2D=TSNEImpl(full_embeddings)
    PLOTImpl(full_embeddings_2D,full_labels,name_file+'Caso2',dim=15)
    
    #######
    print('Caso 3')
    #Caso4 -> KMEANS addestrato su full embeddings (train + test embeddings) e full label (train + test dai file) 
    #         eseguito sull'embeddings intero prodotto dal modello
    #         Plot del embeddings intero prodotto dal modello con le label predette dal kmeans

    indexEmbeddingsFromModel,embeddingsFromModel=CreoEmbFromDictEmb(emb_dict)

    labels_model_predette=KMEANSImpl(full_embeddings, full_labels,embeddingsFromModel,n_clusters)

    embeddingsFromModel_2D=TSNEImpl(embeddingsFromModel)
    PLOTImpl(embeddingsFromModel_2D,labels_model_predette,name_file+'Caso3.png',dim=2)

    ###### MODULARITY ######
    print('Calcolo Modularity')
    listafrozenset=fromIndexLabelCreateCommunityNetworkX(indexEmbeddingsFromModel,labels_model_predette)
    G=creoGraphNetworkX(emb_dict)
    modularity = nxcom.modularity(G,listafrozenset)
    f = open("/content/drive/MyDrive/Colab Notebooks/HNE-master/Evaluate/"+dataset+"_Modularity.txt", "a+")
    f.write("Model: "+str(name_model)+", Dataset: "+str(dataset)+", Attributed: "+str(attributed)+", Supervised: "+str(supervised)+"\n")
    f.write('Modularity: ')
    f.write(str(modularity))
    f.write("\n\n")
    f.close()
    print('modularity')
    print(modularity)