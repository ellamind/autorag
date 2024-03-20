import os
from typing import Dict

from llama_index.core import VectorStoreIndex, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.litellm import LiteLLM
from llama_index.llms.openai import OpenAI
from llama_index.core.storage import StorageContext
from loguru import logger
from dotenv import load_dotenv

from autorag.config import EMBEDDING_MODEL_CONFIGS
from autorag.dataset import get_nodes

load_dotenv()
PERSIST_DIR = "index/"


def get_groovy(config: Dict):
    """
    Get ready to groove with the grooviest LLMs in town!

    This function takes you on a wild ride through the discotheques of AI, 
    where the LLMs wear bell-bottoms and the APIs throw glitter. 
    Whether you're doing the hustle with DiscoLM or boogying down with GPT, 
    this function ensures your AI has got the moves.

    Parameters:
    - config (Dict): A configuration dictionary that includes the LLM ID.

    Returns:
    - llm: A LiteLLM or OpenAI instance, depending on the LLM ID, 
           ready to dance the night away (or just do some language tasks).
    """
    llm_id = config["llm_id"]

    #TODO: DiscoLM here
    if "DISCO" in llm_id.upper():
        logger.info("FEEL THE GROOVE BABY!")
        logger.info("💃🪩🕺"*50)
        llm = LiteLLM(model=llm_id, api_base=os.getenv("ELLAMIND_API_BASE"), api_key=os.getenv("ELLAMIND_API_KEY"), max_tokens=8192)
    else:
        logger.info("Meh!")
        llm = OpenAI(model=llm_id)

    return llm

def get_retriever(config: Dict):
    embedding_id = config["embedding_id"]
    embedding_config = EMBEDDING_MODEL_CONFIGS[embedding_id]
    reranker_id = config["reranker_id"]

    logger.info(f"Setting up {embedding_id=}")
    Settings.llm = get_groovy(config)
    Settings.device = "cpu"
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=embedding_id, 
        embed_batch_size=4, 
        query_instruction=embedding_config["query_instruction"], 
        text_instruction=embedding_config["text_instruction"], 
        pooling=embedding_config["pooling"],
        trust_remote_code=embedding_config["trust_remote_code"],
    )

    persist_dir = f"{PERSIST_DIR}{embedding_id}"
    
    if not os.path.exists(persist_dir):
        logger.info(f"Computing index and persisting to {persist_dir=}")
        os.makedirs(persist_dir, exist_ok=True)
        nodes = get_nodes()
        vector_index = VectorStoreIndex(nodes, show_progress=True)
        vector_index.storage_context.persist(persist_dir=persist_dir)
    else:
        logger.info(f"Loading index from {persist_dir=}")
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        vector_index = load_index_from_storage(storage_context, use_async=True)

    logger.info("Setting up retriever")
    retriever = vector_index.as_retriever(similarity_top_k=config["retriever_top_k"])
    logger.info(f"Setting up {reranker_id=} reranker")

    return retriever

def get_query_engine(retriever, node_postprocessors):
    return RetrieverQueryEngine.from_args(
        retriever, node_postprocessors=node_postprocessors
    )
