# raptor/__init__.py
from .cluster_tree_builder import ClusterTreeBuilder, ClusterTreeConfig
from .EmbeddingModels import (BaseEmbeddingModel, OpenAIEmbeddingModel,
                              SBertEmbeddingModel)
from .RetrievalAugmentation import (RetrievalAugmentation,
                                    RetrievalAugmentationConfig)
from .Retrievers import BaseRetriever
from .SummarizationModels import (BaseSummarizationModel,
                                  GPT3SummarizationModel,
                                  GPT3TurboSummarizationModel)
from .tree_builder import TreeBuilder, TreeBuilderConfig
from .tree_retriever import TreeRetriever, TreeRetrieverConfig
from .tree_structures import Node, Tree

# Heavy optional deps — import on demand
def __getattr__(name):
    if name in ("FaissRetriever", "FaissRetrieverConfig"):
        from .FaissRetriever import FaissRetriever, FaissRetrieverConfig
        return {"FaissRetriever": FaissRetriever, "FaissRetrieverConfig": FaissRetrieverConfig}[name]
    if name in ("BaseQAModel", "GPT3QAModel", "GPT3TurboQAModel", "GPT4QAModel", "UnifiedQAModel"):
        from .QAModels import BaseQAModel, GPT3QAModel, GPT3TurboQAModel, GPT4QAModel, UnifiedQAModel
        return {"BaseQAModel": BaseQAModel, "GPT3QAModel": GPT3QAModel,
                "GPT3TurboQAModel": GPT3TurboQAModel, "GPT4QAModel": GPT4QAModel,
                "UnifiedQAModel": UnifiedQAModel}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
