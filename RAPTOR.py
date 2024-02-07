import logging

from .tree_builder import TreeBuilderConfig, TreeBuilder
from .tree_retriever import TreeRetrieverConfig, TreeRetriever
from .QAModels import BaseQAModel, GPT3TurboQAModel
from .tree_structures import Tree, Node
from .cluster_tree_builder import ClusterTreeConfig, ClusterTreeBuilder

# Define a dictionary to map supported tree builders to their respective configs
supported_tree_builders = {
    "default": (TreeBuilder, TreeBuilderConfig),
    "cluster": (ClusterTreeBuilder, ClusterTreeConfig),
    # Add more tree builders and their configs here
}

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class RetrievalAugmentationConfig:
    def __init__(
        self, 
        tree_builder_config=None, 
        tree_retriever_config=TreeRetrieverConfig(), 
        qa_model=None,
        tree_builder_type="cluster"
    ):
        if tree_builder_type not in supported_tree_builders:
            raise ValueError(f"tree_builder_type must be one of {list(supported_tree_builders.keys())}")

        if not isinstance(tree_retriever_config, TreeRetrieverConfig):
            raise ValueError("tree_retriever_config must be an instance of TreeRetrieverConfig")

        if qa_model is not None and not isinstance(qa_model, BaseQAModel):
            raise ValueError("qa_model must be an instance of BaseQAModel")

        tree_builder_class, tree_builder_config_class = supported_tree_builders[tree_builder_type]

        if tree_builder_config is None:
            tree_builder_config = tree_builder_config_class()

        elif not type(tree_builder_config) is tree_builder_config_class:
            raise ValueError(f"tree_builder_config must be a direct instance of {tree_builder_config_class} for tree_builder_type '{tree_builder_type}'")

        self.tree_builder_config = tree_builder_config
        self.tree_retriever_config = tree_retriever_config
        self.qa_model = qa_model or GPT3TurboQAModel()
        self.tree_builder_type = tree_builder_type

    def log_config(self):
        config_summary = """
        RetrievalAugmentationConfig:
            {tree_builder_config}
            
            {tree_retriever_config}
            
            QA Model: {qa_model}
            Tree Builder Type: {tree_builder_type}
        """.format(
            tree_builder_config=self.tree_builder_config.log_config(),
            tree_retriever_config=self.tree_retriever_config.log_config(),
            qa_model=self.qa_model,
            tree_builder_type=self.tree_builder_type,
        )
        return config_summary

class RetrievalAugmentation:
    """
    A Retrieval Augmentation class that combines the TreeBuilder and TreeRetriever classes.
    Enables adding documents to the tree, retrieving information, and answering questions.
    """

    def __init__(self, config=RetrievalAugmentationConfig(), tree=None):
        """
        Initializes a RetrievalAugmentation instance with the specified configuration.
        Args:
            config (RetrievalAugmentationConfig): The configuration for the RetrievalAugmentation instance.
        """
        if not isinstance(config, RetrievalAugmentationConfig):
            raise ValueError("config must be an instance of RetrievalAugmentationConfig")

        if tree is not None and not isinstance(tree, Tree):
            raise ValueError("tree must be an instance of Tree or None")

        self.tree = tree
        tree_builder_class = supported_tree_builders[config.tree_builder_type][0]
        self.tree_builder = tree_builder_class(config.tree_builder_config)

        self.tree_retriever_config = config.tree_retriever_config
        self.qa_model = config.qa_model

        if self.tree is not None:
            self.retriever = TreeRetriever(self.tree_retriever_config, self.tree)
        else:
            self.retriever = None

        logging.info(f"Successfully initialized RAPTOR with Config {config.log_config()}")


    def add_documents(self, docs):
        """
        Adds documents to the tree and creates a TreeRetriever instance.

        Args:
            docs (str): The input text to add to the tree.
        """
        if self.tree is not None:
            user_input = input("Warning: Overwriting existing tree. Did you mean to call 'add_to_existing' instead? (y/n): ")
            if user_input.lower() == 'y':
                #self.add_to_existing(docs)
                return

        logging.info(f"NARRQA RAPTOR")
        self.tree = self.tree_builder.build_from_text(text=docs)
        self.retriever = TreeRetriever(self.tree_retriever_config, self.tree)


    def retrieve(self, question, start_layer: int = None, num_layers: int = None, max_tokens: int = 3500, collapse_tree: bool = False, return_layer_information: bool = True):
        """
        Retrieves information and answers a question using the TreeRetriever instance.

        Args:
            question (str): The question to answer.
            start_layer (int): The layer to start from. Defaults to self.start_layer.
            num_layers (int): The number of layers to traverse. Defaults to self.num_layers.
            max_tokens (int): The maximum number of tokens. Defaults to 3500.
            use_all_information (bool): Whether to retrieve information from all nodes. Defaults to False.

        Returns:
            str: The context from which the answer can be found.

        Raises:
            ValueError: If the TreeRetriever instance has not been initialized.
        """
        if self.retriever is None:
            raise ValueError("The TreeRetriever instance has not been initialized. Call 'add_documents' first.")

        return self.retriever.retrieve(question, start_layer, num_layers, max_tokens, collapse_tree, return_layer_information)

    def answer_question(self, question, start_layer: int = None, num_layers: int = None, max_tokens: int = 3500, collapse_tree: bool = False, return_layer_information: bool = True):
        """
        Retrieves information and answers a question using the TreeRetriever instance.

        Args:
            question (str): The question to answer.
            start_layer (int): The layer to start from. Defaults to self.start_layer.
            num_layers (int): The number of layers to traverse. Defaults to self.num_layers.
            max_tokens (int): The maximum number of tokens. Defaults to 3500.
            use_all_information (bool): Whether to retrieve information from all nodes. Defaults to False.

        Returns:
            str: The answer to the question.

        Raises:
            ValueError: If the TreeRetriever instance has not been initialized.
        """
        if return_layer_information:
            context, layer_information = self.retrieve(question, start_layer, num_layers, max_tokens, collapse_tree, return_layer_information)
        
        answer = self.qa_model.answer_question(context, question)

        if return_layer_information:
            return answer, layer_information
            
        return answer
