import numpy as np
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.spatial.distance import squareform
from oslo_lib import bootstrap_T

def get_consensus_matrix(n_components, 
                         factorization, 
                         T, 
                         nruns=100, 
                         init='svd', 
                         bootstrap=False, 
                         frac=.8, 
                         data=None, 
                         clusters=None, 
                         use_W=True):
    
    As = []
    
    for run in range(nruns):
        if bootstrap:
            T_ = bootstrap_T(T, frac, data, clusters)
        else:
            T_ = T
        temporal_factors, W, H = factorization(T_, rank=n_components, init=init).factors
        if use_W:
            As.append(get_adjacency_matrix(W))
        else:
            As.append(get_adjacency_matrix(H))
        
    return np.array(As, dtype=float).mean(axis=0)

def get_adjacency_matrix(W):
    NMF_clustering = get_NMF_clustering(W)
    l = len(NMF_clustering)
    A = np.zeros((l,l))
    for i in range(l):
        for j in range(l):
            if NMF_clustering[i]==NMF_clustering[j]:
                A[i,j] = 1
    return A

def get_NMF_clustering(W):
    return {i:get_max_index(w) for i, w in enumerate(W)}

def get_max_index(a):
    return np.where(a==a.max())[0][0]

def get_CCC(C, D):
    
    # https://www.pnas.org/doi/10.1073/pnas.0308531101#sec-1
    # https://uk.mathworks.com/help/stats/cophenet.html

    C_off = np.identity(len(C)) - C
    
    avg_c = C_off.mean()
    avg_d = D.mean()
    
    n_clusters = len(C_off)
    num = 0
    den_1 = 0
    den_2 = 0
    for i in range(n_clusters):
        for j in range(i, n_clusters):
            c = C_off[i, j]
            d = D[i, j]
            num += (c - avg_c) * (d - avg_d)
            den_1 += (c - avg_c)**2
            den_2 += (d - avg_d)**2
                
    den = (den_1 * den_2)**.5
    
    return num/den

def compute_rho(n_components, 
                factorization, 
                T, 
                nruns=100, 
                init='svd', 
                bootstrap=False, 
                frac=.8, 
                data=None, 
                clusters=None, 
                use_W=True):
    C = get_consensus_matrix(n_components, 
                             factorization, 
                             T, 
                             nruns, 
                             init=init, 
                             bootstrap=bootstrap, 
                             frac=frac, 
                             data=data, 
                             clusters=clusters, 
                             use_W=use_W)
    L = linkage(C, 'average')
    coph_distances = squareform(cophenet(L)) 
    rho = get_CCC(C, coph_distances)
    return rho