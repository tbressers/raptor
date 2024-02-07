from typing import Dict, List, Set

import numpy as np
import tiktoken

from sklearn.cluster import SpectralClustering
from scipy import spatial

import logging
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

from .utils import get_embeddings
from .tree_structures import Node

from scipy.spatial.distance import cosine

from sklearn.mixture import GaussianMixture

import umap
import hdbscan

import random 

random.seed(224)

# reduce embedding to 5D using UMAP with high local 
def global_cluster_embeddings(embeddings, dim):
    reduced_embeddings = umap.UMAP(n_neighbors=int((len(embeddings) - 1)**(1/2)), 
                                   n_components=dim, 
                                   metric='cosine').fit_transform(embeddings)
    return reduced_embeddings


def local_cluster_embeddings(embeddings, dim, num_neighbors=10):
    reduced_embeddings = umap.UMAP(n_neighbors=num_neighbors, 
                                   n_components=dim, 
                                   metric='cosine').fit_transform(embeddings)
    return reduced_embeddings


def get_optimal_clusters(embeddings):
    
    max_ = min(50, len(embeddings)) 
    n_clusters = np.arange(1, max_)

    # Calculate the BIC for each number of clusters
    bics = []
    for n in n_clusters:
        gm = GaussianMixture(n_components=n, random_state=0)
        gm.fit(embeddings)
        bics.append(gm.bic(embeddings))

    optimal_clusters = n_clusters[np.argmin(bics)]
    return optimal_clusters

def GMM_cluster(embeddings, threshold):
    
    n_clusters = get_optimal_clusters(embeddings)

    logging.info(
                f"Optimal Clusters Found: {n_clusters}"
            )

    
    gm = GaussianMixture(n_components=n_clusters)
    gm.fit(embeddings)

    # Get the probabilities of each point belonging to each cluster
    probs = gm.predict_proba(embeddings)
    # For each point, find the clusters where the probability is above a certain threshold
    labels = [np.where(prob > threshold)[0] for prob in probs]
    
    return labels, n_clusters


def perform_clustering(embeddings, dim, threshold):
    # First, perform global clustering
 
    reduced_embeddings_global = global_cluster_embeddings(embeddings, dim=dim)
    global_clusters, n_global_clusters = GMM_cluster(reduced_embeddings_global, threshold)

    logging.info(
                f"Global Clusters for Layer: {n_global_clusters}"
            )

    # Initialize a list to store all local clusters
    all_local_clusters = [np.array([]) for _ in range(len(embeddings))]

    # Initialize a counter for the total number of clusters
    total_clusters = 0

    # Now perform local clustering on each of the global clusters
    for i in range(n_global_clusters):
        # Get the embeddings for this global cluster
        global_cluster_embeddings_ = embeddings[np.array([i in gc for gc in global_clusters])]

        logging.info(
                f"Number of Nodes in Global Cluster {i}: {len(global_cluster_embeddings_)}"
            )

        # Skip this global cluster if it has no samples
        if len(global_cluster_embeddings_) == 0:
            continue

        if len(global_cluster_embeddings_) <= dim + 1:
            local_clusters = [np.array([0]) for _ in range(len(global_cluster_embeddings_))]
            n_local_clusters = 1
            
        else:
            # Perform local clustering on these embeddings
            reduced_embeddings_local = local_cluster_embeddings(global_cluster_embeddings_, dim=dim)
            local_clusters, n_local_clusters = GMM_cluster(reduced_embeddings_local, t)

        logging.info(
                f"Number of Local Clusters in Global Cluster {i}: {n_local_clusters}"
            )

        # Add the local clusters to our list, adding the total number of clusters so far to the labels
        for j in range(n_local_clusters):
            local_cluster_embeddings_ = global_cluster_embeddings_[np.array([j in lc for lc in local_clusters])]
            indices = np.where((embeddings == local_cluster_embeddings_[:, None]).all(-1))[1]
            all_local_clusters = [np.append(all_local_clusters[idx], j + total_clusters) if idx in indices else all_local_clusters[idx] for idx in range(len(embeddings))]

        # Update the total number of clusters
        total_clusters += n_local_clusters

    logging.info(
                f"Total Clusters for Layer: {total_clusters}"
            )
    return all_local_clusters


def RAPTOR_Clustering(nodes: List[Node], embedding_model_name: str, max_length_in_cluster=3500, tokenizer=tiktoken.get_encoding("cl100k_base"), reduction_dimension: int = 10, threshold=0.1) -> List[List[Node]]:
    # Get the embeddings from the nodes
    embeddings = np.array([node.embeddings[embedding_model_name] for node in nodes])

    # Perform the clustering
    clusters = perform_clustering(embeddings, dim=reduction_dimension, threshold=threshold)

    # Initialize an empty list to store the clusters of nodes
    node_clusters = []

    # Iterate over each unique label in the clusters
    for label in np.unique(np.concatenate(clusters)):
        # Get the indices of the nodes that belong to this cluster
        indices = [i for i, cluster in enumerate(clusters) if label in cluster]
        
        # Add the corresponding nodes to the node_clusters list
        cluster_nodes = [nodes[i] for i in indices]

        # Base case: if the cluster only has one node, do not attempt to recluster it
        if len(cluster_nodes) == 1:
            node_clusters.append(cluster_nodes)
            continue

        # Calculate the total length of the text in the nodes
        total_length = sum([len(tokenizer.encode(node.text)) for node in cluster_nodes])

        # If the total length exceeds the maximum allowed length, recluster this cluster
        if total_length > max_length_in_cluster:
            logging.info(f'reclustering cluster with {len(cluster_nodes)} nodes')
            node_clusters.extend(RAPTOR_Clustering(cluster_nodes, embedding_model_name, max_length_in_cluster))
        else:
            node_clusters.append(cluster_nodes)

    return node_clusters






