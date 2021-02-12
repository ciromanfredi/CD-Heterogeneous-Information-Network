import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, normalized_mutual_info_score, adjusted_rand_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.cluster import KMeans
import networkx as nx
import networkx.algorithms.community as nxcom
from sklearn.neural_network import MLPClassifier
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

def writeScore(name_model,dataset,attributed,supervised,macro,micro,accuracy,precision,recall,size=None):
  f = open("/content/drive/MyDrive/Colab Notebooks/HNE-master/Evaluate/"+dataset+"_Score.txt", "a+")
  f.write("Model: "+str(name_model)+", Dataset: "+str(dataset)+", Attributed: "+str(attributed)+", Supervised: "+str(supervised))
  if size:
      f.write("Train_Size: "+str(1-size)+" Test_Size: "+str(size))
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

def SVCImpl(train_embeddings, train_labels,test_embeddings,test_labels,seed,max_iter,single_label=True):
    clf = LinearSVC(random_state=seed, max_iter=max_iter)
    clf.fit(train_embeddings, train_labels)
    predssvc = clf.predict(test_embeddings)
    if single_label==True:
        macrosvc,microsvc,accuracysvc,precisionsvc,recallsvc=[],[],[],[],[]
        macrosvc.append(f1_score(test_labels, predssvc, average='macro'))
        microsvc.append(f1_score(test_labels, predssvc, average='micro'))
        accuracysvc.append(accuracy_score(test_labels,predssvc))
        precisionsvc.append(precision_score(test_labels, predssvc, average="macro"))
        recallsvc.append(recall_score(test_labels, predssvc, average="macro"))
        return predssvc, macrosvc, microsvc, accuracysvc, precisionsvc, recallsvc
    else:
        predssvc, scoressvc, accuracylistsvc, precisionlistsvc, recalllistsvc=[], [], [], [], []
        scoressvc.append(f1_score(test_labels, predssvc, average='binary'))
        accuracylistsvc.append(accuracy_score(test_labels,predssvc))
        precisionlistsvc.append(precision_score(test_labels, predssvc, average="binary"))
        recalllistsvc.append(recall_score(test_labels, predssvc, average="binary"))
        return predssvc, scoressvc, accuracylistsvc, precisionlistsvc, recalllistsvc

def LogRegImpl(train_embeddings, train_labels,test_embeddings,test_labels,seed,max_iter,single_label=True):
    
    logreg = LogisticRegression(random_state=seed, max_iter=max_iter)
    logreg.fit(train_embeddings, train_labels)
    predslogreg = logreg.predict(test_embeddings)
    if single_label==True:
        macrologreg,micrologreg,accuracylogreg,precisionlogreg,recalllogreg=[],[],[],[],[]
        macrologreg.append(f1_score(test_labels, predslogreg, average='macro'))
        micrologreg.append(f1_score(test_labels, predslogreg, average='micro'))
        accuracylogreg.append(accuracy_score(test_labels,predslogreg))
        precisionlogreg.append(precision_score(test_labels, predslogreg, average="macro"))
        recalllogreg.append(recall_score(test_labels, predslogreg, average="macro"))
        return predslogreg,macrologreg,micrologreg,accuracylogreg,precisionlogreg,recalllogreg
    else:
        scoreslogreg,accuracylistlogreg,precisionlistlogreg,recalllistlogreg = [], [], [], []
        scoreslogreg.append(f1_score(test_labels, predslogreg, average='binary'))
        accuracylistlogreg.append(accuracy_score(test_labels,predslogreg))
        precisionlistlogreg.append(precision_score(test_labels, predslogreg, average="binary"))
        recalllistlogreg.append(recall_score(test_labels, predslogreg, average="binary"))
        return scoreslogreg,accuracylistlogreg,precisionlistlogreg,recalllistlogreg

def RFImpl(train_embeddings, train_labels,test_embeddings,test_labels,seed,max_iter,single_label=True):
    
    rf = RandomForestClassifier(random_state=seed, max_iter=max_iter)
    rf.fit(train_embeddings, train_labels)
    predsrf = rf.predict(test_embeddings)
    if single_label==True:
        macrorf,microrf,accuracyrf,precisionrf,recallrf=[],[],[],[],[]
        macrorf.append(f1_score(test_labels, predsrf, average='macro'))
        microrf.append(f1_score(test_labels, predsrf, average='micro'))
        accuracyrf.append(accuracy_score(test_labels,predsrf))
        precisionrf.append(precision_score(test_labels, predsrf, average="macro"))
        recallrf.append(recall_score(test_labels, predsrf, average="macro"))
        return predsrf,macrorf,microrf,accuracyrf,precisionrf,recallrf
    else:
        scoresrf,accuracylistrf,precisionlistrf,recalllistrf = [], [], [], []
        scoresrf.append(f1_score(test_labels, predsrf, average='binary'))
        accuracylistrf.append(accuracy_score(test_labels,predsrf))
        precisionlistrf.append(precision_score(test_labels, predsrf, average="binary"))
        recalllistrf.append(recall_score(test_labels, predsrf, average="binary"))
        return scoresrf,accuracylistrf,precisionlistrf,recalllistrf

def ADAMImpl(train_embeddings, train_labels,test_embeddings,test_labels,seed,max_iter,single_label=True):
    
    adam=MLPClassifier(max_iter=max_iter)
    adam.fit(train_embeddings, train_labels)
    predsadam = adam.predict(test_embeddings)
    if single_label==True:
        macroadam,microadam,accuracyadam,precisionadam,recalladam=[],[],[],[],[]
        macroadam.append(f1_score(test_labels, predsadam, average='macro'))
        microadam.append(f1_score(test_labels, predsadam, average='micro'))
        accuracyadam.append(accuracy_score(test_labels,predsadam))
        precisionadam.append(precision_score(test_labels, predsadam, average="macro"))
        recalladam.append(recall_score(test_labels, predsadam, average="macro"))
        return predsadam,macroadam,microadam,accuracyadam,precisionadam,recalladam
    else:
        scoresadam,accuracylistadam,precisionlistadam,recalllistadam=[],[],[],[]
        scoresadam.append(f1_score(test_labels, predsadam, average='binary'))
        accuracylistadam.append(accuracy_score(test_labels,predsadam))
        precisionlistadam.append(precision_score(test_labels, predsadam, average="binary"))
        recalllistadam.append(recall_score(test_labels, predsadam, average="binary"))
        return scoresadam,accuracylistadam,precisionlistadam,recalllistadam

def ConfusionMatrixIMPL(test_labels,preds,txt="Confusion Matrix"):
    cm = confusion_matrix(test_labels,preds)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(txt)
    print(cm)   

def unsupervised_single_class_single_label(full_embeddings, full_labels,n_splits=5,seed=21,max_iter=10000):    
    print("Start: unsupervised_single_class_single_label")
    macrosvc,microsvc,accuracysvc,precisionsvc,recallsvc=[],[],[],[],[]
    macrologreg,micrologreg,accuracylogreg,precisionlogreg,recalllogreg=[],[],[],[],[]
    macrorf,microrf,accuracyrf,precisionrf,recallrf=[],[],[],[],[]
    macroadam,microadam,accuracyadam,precisionadam,recalladam=[],[],[],[],[]
    
    skf = StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=seed)
    index=0
    for train_idx, test_idx in skf.split(full_embeddings, full_labels):

        predssvc, macrosvc, microsvc, accuracysvc, precisionsvc, recallsvc=SVCImpl(full_embeddings[train_idx], full_labels[train_idx],full_embeddings[test_idx],full_labels[test_idx],seed,max_iter)
        predslogreg,macrologreg,micrologreg,accuracylogreg,precisionlogreg,recalllogreg=LogRegImpl(full_embeddings[train_idx], full_labels[train_idx],full_embeddings[test_idx],full_labels[test_idx],seed,max_iter)
        predsrf,macrorf,microrf,accuracyrf,precisionrf,recallrf=RFImpl(full_embeddings[train_idx], full_labels[train_idx],full_embeddings[test_idx],full_labels[test_idx],seed,max_iter)
        predsadam,macroadam,microadam,accuracyadam,precisionadam,recalladam=ADAMImpl(full_embeddings[train_idx], full_labels[train_idx],full_embeddings[test_idx],full_labels[test_idx],seed,max_iter)

        if index==0:
            ConfusionMatrixIMPL(full_labels[test_idx],predssvc,"Confusion Matrix Support Vector Classifier")
            ConfusionMatrixIMPL(full_labels[test_idx],predslogreg,"Confusion Matrix Logistic Regression")
            ConfusionMatrixIMPL(full_labels[test_idx],predsrf,"Confusion Matrix Random Forest")
            ConfusionMatrixIMPL(full_labels[test_idx],predsadam,"Confusion Matrix MLP (ADAM optmizer)")

    writeScore(name_model,dataset,attributed,supervised,macrosvc,microsvc,accuracysvc,precisionsvc,recallsvc)   
    writeScore(name_model,dataset,attributed,supervised,macrologreg,micrologreg,accuracylogreg,precisionlogreg,recalllogreg)   
    writeScore(name_model,dataset,attributed,supervised,macrorf,microrf,accuracyrf,precisionrf,recallrf)   
    writeScore(name_model,dataset,attributed,supervised,macroadam,microadam,accuracyadam,precisionadam,recalladam)   
    #return np.mean(micro), np.mean(macro), np.mean(accuracy), np.mean(precision), np.mean(recall)

def semisupervised_single_class_single_label(train_embeddings, train_labels, test_embeddings,test_labels,num_rip=1,seed=21, max_iter=100000):  
    print("Start: semisupervised_single_class_single_label")
    macrosvc,microsvc,accuracysvc,precisionsvc,recallsvc=[],[],[],[],[]
    macrologreg,micrologreg,accuracylogreg,precisionlogreg,recalllogreg=[],[],[],[],[]
    macrorf,microrf,accuracyrf,precisionrf,recallrf=[],[],[],[],[]
    macroadam,microadam,accuracyadam,precisionadam,recalladam=[],[],[],[],[]

    for _ in range(num_rip):
        
        predssvc, macrosvc, microsvc, accuracysvc, precisionsvc, recallsvc=SVCImpl(train_embeddings, train_labels,test_embeddings,test_labels,seed,max_iter)
        predslogreg,macrologreg,micrologreg,accuracylogreg,precisionlogreg,recalllogreg=LogRegImpl(train_embeddings, train_labels,test_embeddings,test_labels,seed,max_iter)
        predsrf,macrorf,microrf,accuracyrf,precisionrf,recallrf=RFImpl(train_embeddings, train_labels,test_embeddings,test_labels,seed,max_iter)
        predsadam,macroadam,microadam,accuracyadam,precisionadam,recalladam=ADAMImpl(train_embeddings, train_labels,test_embeddings,test_labels,seed,max_iter)
        
        if _ == 0:
            ConfusionMatrixIMPL(test_labels,predssvc,"Confusion Matrix Support Vector Classifier")
            ConfusionMatrixIMPL(test_labels,predslogreg,"Confusion Matrix Logistic Regression")
            ConfusionMatrixIMPL(test_labels,predsrf,"Confusion Matrix Random Forest")
            ConfusionMatrixIMPL(test_labels,predsadam,"Confusion Matrix MLP (ADAM optmizer)")

    writeScore(name_model,dataset,attributed,supervised,macrosvc,microsvc,accuracysvc,precisionsvc,recallsvc)   
    writeScore(name_model,dataset,attributed,supervised,macrologreg,micrologreg,accuracylogreg,precisionlogreg,recalllogreg)   
    writeScore(name_model,dataset,attributed,supervised,macrorf,microrf,accuracyrf,precisionrf,recallrf)   
    writeScore(name_model,dataset,attributed,supervised,macroadam,microadam,accuracyadam,precisionadam,recalladam)   
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
    total_scoreadam,accuracylist2adam,precisionlist2adam,recalllist2adam = [], [], [], []
    
    for ntype, binary_label in enumerate(binary_labels):
        
        scoressvc,accuracylistsvc,precisionlistsvc,recalllistsvc = [], [], [], []
        scoreslogreg,accuracylistlogreg,precisionlistlogreg,recalllistlogreg = [], [], [], []
        scoresrf,accuracylistrf,precisionlistrf,recalllistrf = [], [], [], []
        scoresadam,accuracylistadam,precisionlistadam,recalllistadam = [], [], [], []

        skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=seed)
        for train_idx, test_idx in skf.split(embs, binary_label):            

            scoressvc,accuracylistsvc,precisionlistsvc,recalllistsvc=SVCImpl(embs[train_idx], binary_label[train_idx],embs[test_idx],binary_label[test_idx],seed,max_iter,single_label=False)
            scoreslogreg,accuracylistlogreg,precisionlistlogreg,recalllistlogreg=LogRegImpl(embs[train_idx], binary_label[train_idx],embs[test_idx],binary_label[test_idx],seed,max_iter,single_label=False)
            scoresrf,accuracylistrf,precisionlistrf,recalllistrf=RFImpl(embs[train_idx], binary_label[train_idx],embs[test_idx],binary_label[test_idx],seed,max_iter,single_label=False)
            scoresadam,accuracylistadam,precisionlistadam,recalllistadam=ADAMImpl(embs[train_idx], binary_label[train_idx],embs[test_idx],binary_label[test_idx],seed,max_iter,single_label=False)


        weights.append(sum(binary_label))
        total_scoressvc.append(sum(scoressvc)/5)
        accuracylist2svc.append(np.mean(accuracylistsvc))
        precisionlist2svc.append(np.mean(precisionlistsvc))
        recalllist2svc.append(np.mean(recalllistsvc))

        total_scoreslogreg.append(sum(scoreslogreg)/5)
        accuracylist2logreg.append(np.mean(accuracylistlogreg))
        precisionlist2logreg.append(np.mean(precisionlistlogreg))
        recalllist2logreg.append(np.mean(recalllistlogreg))

        total_scoresrf.append(sum(scoresrf)/5)
        accuracylist2rf.append(np.mean(accuracylistrf))
        precisionlist2rf.append(np.mean(precisionlistrf))
        recalllist2rf.append(np.mean(recalllistrf))

        total_scoreadam.append(sum(scoresadam)/5)
        accuracylist2adam.append(np.mean(accuracylistadam))
        precisionlist2adam.append(np.mean(precisionlistadam))
        recalllist2adam.append(np.mean(recalllistadam))

    #macrosvc = sum(total_scoressvc)/len(total_scoressvc)
    microsvc = sum([score*weight for score, weight in zip(total_scoressvc, weights)])/sum(weights)

    #macrologreg = sum(total_scoreslogreg)/len(total_scoreslogreg)
    micrologreg = sum([score*weight for score, weight in zip(total_scoreslogreg, weights)])/sum(weights)

    #macrorf = sum(total_scoresrf)/len(total_scoresrf)
    microrf = sum([score*weight for score, weight in zip(total_scoresrf, weights)])/sum(weights)

    #macrorf = sum(total_scoresrf)/len(total_scoresrf)
    microadam = sum([score*weight for score, weight in zip(total_scoreadam, weights)])/sum(weights)


    writeScore(name_model,dataset,attributed,supervised,total_scoressvc,microsvc,accuracylist2svc,precisionlist2svc,recalllist2svc)    
    writeScore(name_model,dataset,attributed,supervised,total_scoreslogreg,micrologreg,accuracylist2logreg,precisionlist2logreg,recalllist2logreg)    
    writeScore(name_model,dataset,attributed,supervised,total_scoresrf,microrf,accuracylist2rf,precisionlist2rf,recalllist2rf)    
    writeScore(name_model,dataset,attributed,supervised,total_scoreadam,microadam,accuracylist2adam,precisionlist2adam,recalllist2adam)    
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
    scoressvc,accuracylistsvc,precisionlistsvc,recalllistsvc,total_scoressvc = [], [], [], [], []
    scoreslogreg,accuracylistlogreg,precisionlistlogreg,recalllistlogreg, total_scoreslogreg = [], [], [], [], []
    scoresrf,accuracylistrf,precisionlistrf,recalllistrf, total_scoresrf = [], [], [], [], []
    scoreadam,accuracylistadam,precisionlistadam,recalllistadam, total_scoresadam=[],[],[],[],[]


    for ntype, (train_label, test_label) in enumerate(zip(train_labels, test_labels)):           

        scoressvc,accuracylistsvc,precisionlistsvc,recalllistsvc=SVCImpl(train_embs, train_labels,test_embs,test_labels,seed,max_iter,single_label=False)
        scoreslogreg,accuracylistlogreg,precisionlistlogreg,recalllistlogreg=LogRegImpl(train_embs, train_labels,test_embs,test_labels,seed,max_iter,single_label=False)
        scoresrf,accuracylistrf,precisionlistrf,recalllistrf=RFImpl(train_embs, train_labels,test_embs,test_labels,seed,max_iter,single_label=False)
        scoresadam,accuracylistadam,precisionlistadam,recalllistadam=ADAMImpl(train_embs, train_labels,test_embs,test_labels,seed,max_iter,single_label=False)

        weights.append(sum(test_label))
        
        total_scoressvc.append(scoressvc)
        total_scoreslogreg.append(scoreslogreg)
        total_scoresrf.append(scoresrf)
        total_scoresadam.append(scoresadam)
        
    #macrosvc = sum(total_scoressvc)/len(total_scoressvc)
    microsvc = sum([score*weight for score, weight in zip(total_scoressvc, weights)])/sum(weights)

    #macrologreg = sum(total_scoreslogreg)/len(total_scoreslogreg)
    micrologreg = sum([score*weight for score, weight in zip(total_scoreslogreg, weights)])/sum(weights)

    #macrorf = sum(total_scoresrf)/len(total_scoresrf)
    microrf = sum([score*weight for score, weight in zip(total_scoresrf, weights)])/sum(weights)

    #macrorf = sum(total_scoresrf)/len(total_scoresrf)
    microadam = sum([score*weight for score, weight in zip(total_scoresadam, weights)])/sum(weights)

    writeScore(name_model,dataset,attributed,supervised,total_scoressvc,microsvc,accuracylistsvc,precisionlistsvc,recalllistsvc)    
    writeScore(name_model,dataset,attributed,supervised,total_scoreslogreg,micrologreg,accuracylistlogreg,precisionlistlogreg,recalllistlogreg)    
    writeScore(name_model,dataset,attributed,supervised,total_scoresrf,microrf,accuracylistrf,precisionlistrf,recalllistrf)   
    writeScore(name_model,dataset,attributed,supervised,total_scoresadam,microadam,accuracylistadam,precisionlistadam,recalllistadam)    
    #return macro, micro, accuracymean, precisionmean, recallmean

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
            semisupervised_single_class_single_label(train_embeddings, train_labels, test_embeddings,test_labels,num_rip=10)
            semisupervised_single_class_single_label_con_percentuale(train_embeddings, train_labels, test_embeddings,test_labels,num_rip=10)
    elif dataset=="Yelp":
        train_labels,train_embeddings,test_labels,test_embeddings=transformYelpSingleLabel(label_file_path,label_test_path,emb_dict)
        full_embeddings = np.append(train_embeddings,test_embeddings,axis=0)
        full_labels = np.append(train_labels,test_labels,axis=0)
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