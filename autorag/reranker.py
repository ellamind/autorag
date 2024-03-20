from typing import Dict, List

from llama_index.postprocessor.colbert_rerank import ColbertRerank
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.postprocessor.types import BaseNodePostprocessor

from loguru import logger


def get_node_postprocessors(config: Dict) -> List[BaseNodePostprocessor]:
    reranker_top_k = config["reranker_top_k"]
    reranker_id = config["reranker_id"]
    if reranker_id == "ColBERTv2":
        return [ColbertRerank(
            top_n=reranker_top_k,
            model="colbert-ir/colbertv2.0",
            tokenizer="colbert-ir/colbertv2.0",
            keep_retrieval_score=True,
            device="cpu"
        )]
    elif reranker_id == "BAAI/bge-reranker-large":
        return [SentenceTransformerRerank(
            top_n=reranker_top_k,
            model="BAAI/bge-reranker-large",
            keep_retrieval_score=True,
            device="cpu"
        )]

    else:
        logger.info("No reranker specified")
        return []

