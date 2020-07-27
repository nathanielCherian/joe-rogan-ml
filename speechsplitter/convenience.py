from sklearn.manifold import TSNE

def quick_TSNE(X, random_state=420, dim=2):

    X_tsne = TSNE(n_components=dim, random_state=random_state).fit_transform(X)

    return ( X_tsne[:, d] for d in range(dim) )