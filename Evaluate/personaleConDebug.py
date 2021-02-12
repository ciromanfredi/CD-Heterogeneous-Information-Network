import warnings
import numpy as np
from collections import defaultdict
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, normalized_mutual_info_score, adjusted_rand_score, average_precision_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.exceptions import UndefinedMetricWarning, ConvergenceWarning
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from collections import Counter
import random
from sklearn.cluster import KMeans
import networkx as nx
from networkx.generators import community
import networkx.algorithms.community as nxcom

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

#Tra le varie classi associate ad un'istanza prende la prima
def transformYelpSingleLabel(label_file_path,label_test_path,emb_dict,zero=False):
    train_labels, train_embeddings = [], []
    with open(label_file_path,'r') as label_file:
      for line in label_file:
          index, _, _, label = line[:-1].split('\t')
          if not zero:
              estremo = len(label.split(','))-1
              label = label.split(',')[random.randint(0,estremo)]
          else:
              label = label.split(',')[0]
          train_labels.append(label)
          train_embeddings.append(emb_dict[index])    
    train_labels, train_embeddings = np.array(train_labels).astype(int), np.array(train_embeddings)  
  
    test_labels, test_embeddings = [], []
    with open(label_test_path,'r') as label_file:
      for line in label_file:
          index, _, _, label = line[:-1].split('\t')
          if not zero:
              estremo = len(label.split(','))-1
              label = label.split(',')[random.randint(0,estremo)]
          else:
              label = label.split(',')[0]
          test_labels.append(label)
          test_embeddings.append(emb_dict[index])    
    test_labels, test_embeddings = np.array(test_labels).astype(int), np.array(test_embeddings)
    return train_labels,train_embeddings,test_labels,test_embeddings  

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
  #plt.show()
  plt.clf()

def CreoEmbFromDictEmb(emb_dict):
  print('STARTING EMBEDDING CREATION (from dict)')
  #embeddings = np.empty((0,dim_emb),float)
  indexlist, embeddings =[], []
  for key,value in emb_dict.items():
      #embeddings = np.vstack([embeddings, value])
      indexlist.append(key)
      embeddings.append(value)
  indexlist,embeddings=np.array(indexlist).astype(int),np.array(embeddings).astype(np.float32)
  return indexlist,embeddings

def CreoEmbFromPath(emb_file_path):
    print('STARTING EMBEDDING CREATION (from path)')
    #embeddings = np.empty((0,dim_emb),float)
    embeddings=[]
    with open(emb_file_path,'r') as emb_file:        
        for i, line in enumerate(emb_file):
              #print(i)
            if i == 0:
                train_para = line[:-1]
            else:
                index, emb = line[:-1].split('\t')
                #embeddings.append([float(emb.split())])
                embeddings.append([float(x) for x in emb.split()])
                #embeddings = np.vstack([embeddings, np.array(emb.split()).astype(np.float32)])
                #emb_dict[index] = np.array(emb.split()).astype(np.float32)
        embeddings=np.array(embeddings).astype(np.float32)
    return train_para, embeddings

def writeScore(name_model,dataset,attributed,supervised,macro,micro,accuracy,precision,recall):
  f = open("/content/drive/MyDrive/Colab Notebooks/HNE-master/Evaluate/"+dataset+"_Score.txt", "a+")
  f.write("Model: "+str(name_model)+", Dataset: "+str(dataset)+", Attributed: "+str(attributed)+", Supervised: "+str(supervised))
  f.write('\nMacro-F1: ')
  f.write(str(np.mean(macro)))
  f.write(', Micro-F1: ')
  f.write(str(np.mean(micro)))
  f.write('\nAccuracy: ')
  f.write(str(np.mean(accuracy)))
  f.write(', Precision: ')
  f.write(str(np.mean(precision)))
  f.write(', Recall: ')
  f.write(str(np.mean(recall)))
  f.write("\n\n")
  f.close()

def unsupervised_single_class_single_label(full_embeddings, full_labels,n_splits=5,seed=21,max_iter=10000):      
    macro,micro,accuracy,precision,recall=[],[],[],[],[]
    skf = StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=seed)
    for train_idx, test_idx in skf.split(full_embeddings, full_labels):
        #print('train_idx')
        #print(train_idx)
        #print('test_idx')
        #print(test_idx)
        
        clf = LinearSVC(random_state=seed, max_iter=max_iter)
        clf.fit(full_embeddings[train_idx], full_labels[train_idx])
        preds = clf.predict(full_embeddings[test_idx])

        macro.append(f1_score(full_labels[test_idx], preds, average='macro'))
        micro.append(f1_score(full_labels[test_idx], preds, average='micro'))
        accuracy.append(accuracy_score(full_labels[test_idx],preds))
        precision.append(precision_score(full_labels[test_idx], preds, average="macro"))
        recall.append(recall_score(full_labels[test_idx], preds, average="macro"))
    
    writeScore(name_model,dataset,attributed,supervised,np.mean(macro),np.mean(micro),np.mean(accuracy),np.mean(precision),np.mean(recall))    
    return np.mean(micro), np.mean(macro), np.mean(accuracy), np.mean(precision), np.mean(recall)

def semisupervised_single_class_single_label(train_embeddings, train_labels, test_embeddings,test_labels,num_rip=1,seed=21, max_iter=10000):     
    macro,micro,accuracy,precision,recall=[],[],[],[],[]
    for _ in range(num_rip):
        clf = LinearSVC(random_state=seed, max_iter=max_iter)
        clf.fit(train_embeddings, train_labels)
        preds = clf.predict(test_embeddings)  

        macro.append(f1_score(test_labels, preds, average='macro'))
        micro.append(f1_score(test_labels, preds, average='micro'))
        accuracy.append(accuracy_score(test_labels,preds))
        precision.append(precision_score(test_labels, preds, average="macro"))
        recall.append(recall_score(test_labels, preds, average="macro"))

    writeScore(name_model,dataset,attributed,supervised,np.mean(macro),np.mean(micro),np.mean(accuracy),np.mean(precision),np.mean(recall))    
    return np.mean(macro), np.mean(macro), np.mean(accuracy), np.mean(precision), np.mean(recall)

def semisupervised_single_class_multi_label(label_file_path, label_test_path, emb_dict, seed=21, max_iter=3000):

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
    
    weights, total_scores, accuracy, precision, recall = [], [], [], [], [] 
    for ntype, (train_label, test_label) in enumerate(zip(train_labels, test_labels)):           
        
        clf = LinearSVC(random_state=seed, max_iter=max_iter)
        clf.fit(train_embs, train_label)
        preds = clf.predict(test_embs)
        #scores = append(f1_score(test_label, preds, average='binary'))
        scores = (f1_score(test_label, preds, average='binary'))
        accuracy.append(accuracy_score(test_labels,preds))
        precision.append(precision_score(test_labels, preds, average="binary"))
        recall.append(recall_score(test_labels, preds, average="binary"))

        weights.append(sum(test_label))
        total_scores.append(scores)
        
    macro = sum(total_scores)/len(total_scores)
    micro = sum([score*weight for score, weight in zip(total_scores, weights)])/sum(weights)
    accuracymean=np.mean(accuracy)
    precisionmean=np.mean(precision)
    recallmean=np.mean(recall)
    
    writeScore(name_model,dataset,attributed,supervised,macro,micro,accuracymean,precisionmean,recallmean)    
    return macro, micro, accuracymean, precisionmean, recallmean

def unsupervised_single_class_multi_label(label_file_path, label_test_path, emb_dict, seed=21, max_iter=3000):

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
    
    weights, total_scores,accuracylist2,precisionlist2,recalllist2 = [], [], [], [], []
    for ntype, binary_label in enumerate(binary_labels):
        
        scores,accuracylist,precisionlist,recalllist = [], [], [], []
        skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=seed)
        for train_idx, test_idx in skf.split(embs, binary_label):            
        
            clf = LinearSVC(random_state=seed, max_iter=max_iter)
            clf.fit(embs[train_idx], binary_label[train_idx])
            preds = clf.predict(embs[test_idx])
            scores.append(f1_score(binary_label[test_idx], preds, average='binary'))
            accuracylist.append(accuracy_score(binary_label[test_idx],preds))
            precisionlist.append(precision_score(binary_label[test_idx], preds, average="binary"))
            recalllist.append(recall_score(binary_label[test_idx], preds, average="binary"))

        weights.append(sum(binary_label))
        total_scores.append(sum(scores)/5)
        accuracylist2.append(np.mean(accuracylist))
        precisionlist2.append(np.mean(precisionlist))
        recalllist2.append(np.mean(recalllist))
        
    macro = sum(total_scores)/len(total_scores)
    micro = sum([score*weight for score, weight in zip(total_scores, weights)])/sum(weights)
    accuracymean=np.mean(accuracylist2)
    precisionmean=np.mean(precisionlist2)
    recallmean=np.mean(recalllist2)

    writeScore(name_model,dataset,attributed,supervised,macro,micro,accuracymean,precisionmean,recallmean)    
    return macro, micro, accuracymean, precisionmean, recallmean

def writeClusteringScore(name_model,dataset,attributed,supervised,nmi_list,ari_list):
    f = open("/content/drive/MyDrive/Colab Notebooks/HNE-master/Evaluate/"+dataset+"_Score.txt", "a+")
    f.write("Model: "+str(name_model)+", Dataset: "+str(dataset)+", Attributed: "+str(attributed)+", Supervised: "+str(supervised)+"\n")
    f.write('NMI_score: ')
    f.write(str(np.mean(nmi_list))+ ", StdNMI: " + str(np.std(nmi_list)))
    f.write(', ARI_score: ')
    f.write(str(np.mean(ari_list)) +", StdARI: "+ str(np.std(ari_list)))
    f.write("\n\n")
    f.close()

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

def GMMImpl(train_embs,test_embs,n_components,labels_originali=[],num_rip=1):
  from sklearn.mixture import GaussianMixture
  gmm = GaussianMixture(n_components=n_components)
  gmm.fit(train_embs)
  if len(labels_originali) > 0:
      nmi_list, ari_list = [], []
      macro,micro,accuracy,precision,recall=[],[],[],[],[]
      print('len labels originali')
      print(len(labels_originali))
      for _ in range(num_rip):
          labels_predette = GMMImpl(train_embs,test_embs,n_components)
          #print('len labels_predette')
          #print(len(labels_predette))
          nmi_score = normalized_mutual_info_score(labels_originali, labels_predette, average_method='arithmetic')
          ari_score = adjusted_rand_score(labels_originali, labels_predette)
          nmi_list.append(nmi_score)
          ari_list.append(ari_score)
          macro.append(f1_score(labels_originali, labels_predette, average='macro'))
          micro.append(f1_score(labels_originali, labels_predette, average='micro'))
          accuracy.append(accuracy_score(labels_originali,labels_predette))
          precision.append(precision_score(labels_originali, labels_predette, average="macro"))
          recall.append(recall_score(labels_originali, labels_predette, average="macro"))
          print(np.mean(precision))
      print('recall score')
      print(np.mean(recall))
      f = open("/content/drive/MyDrive/Colab Notebooks/HNE-master/Evaluate/"+dataset+"_Score.txt", "a+")
      f.write("Model: "+str(name_model)+", Dataset: "+str(dataset)+", Attributed: "+str(attributed)+", Supervised: "+str(supervised)+"\n")
      f.write('NMI_score: ')
      f.write(str(np.mean(nmi_list))+ ", StdNMI: " + str(np.std(nmi_list)))
      f.write(', ARI_score: ')
      f.write(str(np.mean(ari_list)) +", StdARI: "+ str(np.std(ari_list)))
      f.write("\n\n")
      f.close()
  else:
      #predictions from gmm
      labels = gmm.predict(test_embs)
      #print('[provagmm] labels:')
      #print(labels)

      #probability=gmm.predict_proba(test_embs)
      #print('[provagmm] probability:')
      #print(probability)
      #print('len(probability[0])')
      #print(len(probability[0]))

      #score_samples=gmm.score_samples(test_embs)
      #print('[provagmm] score_samples:')
      #print(score_samples)

      list1 = labels.tolist()
      labels_model_predette=[]
      for index in list1:
          labels_model_predette.append(index)
          #print('index')
          #print(index)
          #print('probability[count,index]')
          #print(probability[count,index])
      return labels_model_predette

def creoGraphNetworkX(emb_dict):
  G = nx.Graph()

  #ADD NODES
  for index,embeddings in emb_dict.items():
      #print('embeddings')
      #print(list(embeddings))
      #attributes=embeddings.split()
      G.add_node(index, attr = list(embeddings))

  archi = open("/content/drive/MyDrive/Colab Notebooks/HNE-master/Data/"+dataset+"/link.dat")
  for arco in archi:
    elementi=str(arco).split('\t')
    if elementi[0] in G.nodes():
      #print("Aggiungo arco tra il nodo: "+str(elementi[0])+" e il nodo: "+str(elementi[1]))
      G.add_edge(elementi[0],elementi[1],type=elementi[2],weight=int(elementi[3]))
    #else:
      #print("La sorgente dell'arco non è presente nei nodi: "+str(elementi[0]))

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
  #print(listafrozenset)
  return listafrozenset

def community_detection_personale(name_model,dataset,attributed,supervised):
    #global name_model
    #global dataset
    #global attributed
    #global supervised

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
    #print('Path file dell'embeddings: ')
    #print(emb_file_path)

    train_para, emb_dict=load(emb_file_path)
    #print('Lunghezza del dizionario: ')
    #print(len(emb_dict))

    print('Calcolo Micro, Macro, accuracy, precision, recall')
    if dataset=="PubMed":
        train_index,test_index,train_labels,train_embeddings,test_labels,test_embeddings=creoTrainTestLabel(label_file_path,label_test_path,emb_dict)
        full_embeddings = np.append(train_embeddings,test_embeddings,axis=0)
        full_labels = np.append(train_labels,test_labels,axis=0)
        if not supervised:
            unsupervised_single_class_single_label(full_embeddings, full_labels)
        else:
            semisupervised_single_class_single_label(train_embeddings, train_labels, test_embeddings,test_labels,num_rip=10)
    elif dataset=="Yelp":
        train_labels,train_embeddings,test_labels,test_embeddings=transformYelpSingleLabel(label_file_path,label_test_path,emb_dict)
        full_embeddings = np.append(train_embeddings,test_embeddings,axis=0)
        full_labels = np.append(train_labels,test_labels,axis=0)
        if not supervised:
            unsupervised_single_class_multi_label(label_file_path, label_test_path, emb_dict)
        else:
            semisupervised_single_class_multi_label(label_file_path, label_test_path, emb_dict)

    
    n_clusters = len(set(full_labels))
    #print('embs and labs full')
    #print(full_embeddings)
    #print(full_labels)
      
    ######
    print('Caso 1')
    #Caso1 -> Kmeans addestrato su train_embeddings - train_labels, testato su test_embeddings
    #         Tsne senza perplexity 
    #         Plot dello spazio embeddings di test con le label predette dal kMeans
    test_labels_predette=KMEANSImpl(train_embeddings, train_labels,test_embeddings,n_clusters)
    test_embeddings_2D=TSNEImpl(test_embeddings)
    #print(Counter(test_labels_predette))
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
    #print('len full_embeddings')
    #print(len(full_embeddings))
    #print('len full_labels')
    #print(len(full_labels))
    indexEmbeddingsFromModel,embeddingsFromModel=CreoEmbFromDictEmb(emb_dict)
    #print('len(indexEmbeddingsFromModel)')
    #print(len(indexEmbeddingsFromModel))
    #if dataset=="PubMed":
    labels_model_predette=KMEANSImpl(full_embeddings, full_labels,embeddingsFromModel,n_clusters)
    #else:
    #    labels_model_predette=GMMImpl(embeddingsFromModel,embeddingsFromModel,n_clusters=n_clusters)
            
    #print('len(labels_model_predette)')
    #print(len(labels_model_predette))
    #print(Counter(labels_model_predette))
    #print('len labels_model_predette')
    #print(len(labels_model_predette))
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