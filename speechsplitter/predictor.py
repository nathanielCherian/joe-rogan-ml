import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from sklearn.mixture import GaussianMixture


def scale_and_pca(X, n_components=0.95):

    st_scaler = StandardScaler()
    X_ss = st_scaler.fit_transform(X)

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_ss)

    return X_pca


def optimal_K(X_pca, max=40):

    inertias = []
    sils = []

    for k in range(2, 40):
        kmeans = KMeans(n_clusters=k).fit(X_pca)
        inertias.append(kmeans.inertia_)
        sils.append(silhouette_score(X_pca, kmeans.labels_))


    return np.argmax(sils) + 2


def gaussian_clustering(K, X_pca, n_init=10, random_state=420):

    gm = GaussianMixture(n_components=K, n_init=n_init, random_state=random_state)
    gm.fit(X_pca)
    y_gm = gm.predict(X_pca)

    return y_gm


def reconstruction_error(pca, X):
    X_pca = pca.transform(X)
    X_recon = pca.inverse_transform(X_pca)
    mse = np.square(X_recon-X).mean(axis=-1)
    return mse





def pca_reconstruction_error(X_pca, y_pred):

    clustered_data = {x:X_pca[y_pred == x] for x in set(y_pred)}
    y_pred = pd.Series(y_pred)
    labels = pd.DataFrame(y_pred, columns=['og'])


    for key, item in clustered_data.items():
        
        cluster_pca = PCA(n_components=0.99).fit(item)
        rec = reconstruction_error(cluster_pca, item)
        threshold = np.std(rec) + rec.mean()
        
        cluster_labels = y_pred.copy()
        
        non_cluster = cluster_labels[cluster_labels != key]
        inds = non_cluster[list(reconstruction_error(cluster_pca, X_pca[cluster_labels != key]) < threshold)].index
        
        cluster_labels.iloc[inds] = key
        #print(f"{key} : {inds}")
        print(f"replaced {len(inds)} instance(s) with {key}")
        
        labels[key] = cluster_labels


    return labels