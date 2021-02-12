import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def kmeans_test(X, y, n_clusters, repeat=10):
    #print(X)
    #print(y)
    #print(n_clusters)
    nmi_list = []
    ari_list = []
    for _ in range(repeat):
        kmeans = KMeans(n_clusters=n_clusters)
        y_pred = kmeans.fit_predict(X)
        print(y_pred)
        nmi_score = normalized_mutual_info_score(y, y_pred, average_method='arithmetic')
        ari_score = adjusted_rand_score(y, y_pred)
        nmi_list.append(nmi_score)
        ari_list.append(ari_score)
    return np.mean(nmi_list), np.std(nmi_list), np.mean(ari_list), np.std(ari_list)

def evaluate_results_nc(embeddings, labels, num_classes):
    print('K-means test')
    nmi_mean, nmi_std, ari_mean, ari_std = kmeans_test(embeddings, labels, num_classes)
    print('NMI: {:.6f}~{:.6f}'.format(nmi_mean, nmi_std))
    print('ARI: {:.6f}~{:.6f}'.format(ari_mean, ari_std))

    return nmi_mean, nmi_std, ari_mean, ari_std

def kmeans_without_test(X, n_clusters):
    #print(X)
    #print(n_clusters)
    kmeans = KMeans(n_clusters=n_clusters)
    y_pred = kmeans.fit_predict(X)
    return y_pred

def PCAImpl(X,num_dim=2):
    pca = PCA(num_dim) 
    pca.fit(X)
    X1=pca.transform(X)
    return X1

def drawImpl(X,label,nomefile):
    plt.scatter(X[:, 0], X[:, 1], c=label ,s=50)
    plt.savefig(nomefile)

def creoSpazio(emb_file_path):
    emb_dict = {}
    with open(emb_file_path,'r') as emb_file:        
        for i, line in enumerate(emb_file):
            if i == 0:
                train_para = line[:-1]
            else:
                index, emb = line[:-1].split('\t')
                emb_dict[index] = np.array(emb.split()).astype(np.float32)
    labels = [] 
    embeddings = np.empty((0,50),float)  
    for file_path in ["/content/drive/MyDrive/Colab Notebooks/HNE-master/Data/PubMed/label.dat", "/content/drive/MyDrive/Colab Notebooks/HNE-master/Data/PubMed/label.dat.test"]:
        with open(file_path,'r') as label_file:
            for line in label_file:
                index, _, _, label = line[:-1].split('\t')
                labels.append(int(label))
                embeddings = np.vstack([embeddings, emb_dict[int(index)]])
    
    return embeddings,labels

def TSNEImpl(embeddings):
    tsne_model = TSNE(n_components=2, init='pca', n_iter=2500, random_state=23)
    new_outputs=tsne_model.fit_transform(embeddings)
    return new_outputs