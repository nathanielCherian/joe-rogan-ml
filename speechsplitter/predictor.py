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

