from typing import Dict, List, Set

from .tree_builder import TreeBuilder, TreeBuilderConfig

from .tree_structures import Tree, Node

from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from .utils import (distances_from_embeddings, get_children, get_embeddings,
                   indices_of_nearest_neighbors_from_distances, get_node_list, get_text,
                   split_text)

from .cluster_utils import spectral_clustering, RAPTOR_Clustering

import logging

import pickle
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class ClusterTreeConfig(TreeBuilderConfig):
    def __init__(
        self,
        reduction_dimension=10,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.reduction_dimension = reduction_dimension

    def log_config(self):
        base_summary = super().log_config()
        cluster_tree_summary = """
        Reduction Dimension: {reduction_dimension}
        """.format(reduction_dimension=self.reduction_dimension)
        return base_summary + cluster_tree_summary



class ClusterTreeBuilder(TreeBuilder):
    def __init__(self, config) -> None:
        super().__init__(config)

        if not isinstance(config, ClusterTreeConfig):
            raise ValueError(f"config must be an instance of ClusterTreeConfig")
        self.reduction_dimension = config.reduction_dimension

        logging.info(f"Successfully initialized ClusterTreeBuilder with Config {config.log_config()}")



    def construct_tree(
        self, current_level_nodes: Dict[int, Node], all_tree_nodes: Dict[int, Node], layer_to_nodes: Dict[int, List[Node]], use_multithreading: bool = False
    ) -> Dict[int, Node]:
        """
        Constructs the hierarchical tree structure layer by layer by iteratively summarizing groups
        of relevant nodes and updating the current_level_nodes and all_tree_nodes dictionaries at each step.

        Args:
            current_level_nodes (Dict[int, Node]): The current set of nodes.
            all_tree_nodes (Dict[int, Node]): The dictionary of all nodes.

        Returns:
            Dict[int, Node]: The final set of root nodes.
        """
        logging.info("Using Cluster TreeBuilder")

        next_node_index = len(all_tree_nodes)

        def process_cluster(cluster, new_level_nodes, next_node_index, summarization_length, lock):
            node_texts = get_text(cluster)

            summarized_text = self.summarize(
                context=node_texts,
                max_tokens=summarization_length,
            )

            logging.info(
                f"Node Texts Length: {len(self.tokenizer.encode(node_texts))}, Summarized Text Length: {len(self.tokenizer.encode(summarized_text))}"
            )

            __, new_parent_node = self.create_node(
                next_node_index,
                summarized_text,
                {node.index for node in cluster}
            )

            with lock:
                new_level_nodes[next_node_index] = new_parent_node

        for layer in range(self.num_layers):

            new_level_nodes = {}

            logging.info(f"Constructing Layer {layer}")

            node_list_current_layer = get_node_list(current_level_nodes)

            if len(node_list_current_layer) <= self.reduction_dimension + 1:
                self.num_layers = layer
                logging.info(f"Stopping Layer construction: Cannot Create More Layers. Total Layers in tree: {layer}")
                break

            clusters = RAPTOR_Clustering(node_list_current_layer, self.cluster_embedding_model, reduction_dimension=self.reduction_dimension)

            lock = Lock()

            summarization_length = self.summarization_length 
            logging.info(f"Summarization Length: {summarization_length}")

            if use_multithreading:
                with ThreadPoolExecutor() as executor:
                    for cluster in clusters:
                        executor.submit(process_cluster, cluster, new_level_nodes, next_node_index, summarization_length, lock)
                        next_node_index += 1
                    executor.shutdown(wait=True)

            else:
                for cluster in clusters:
                    process_cluster(cluster, new_level_nodes, next_node_index, summarization_length, lock)
                    next_node_index += 1

            layer_to_nodes[layer + 1] = list(new_level_nodes.values())
            current_level_nodes = new_level_nodes
            all_tree_nodes.update(new_level_nodes)

            tree = Tree(all_tree_nodes, layer_to_nodes[layer + 1], layer_to_nodes[0], layer + 1, layer_to_nodes)

        return current_level_nodes


