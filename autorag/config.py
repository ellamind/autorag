SWEEP_CONFIG = {
    "method": "grid",
    "metric": {"name": "faithfulness", "goal": "maximize"},
    "metric": {"name": "MRR", "goal": "maximize"},
    "parameters": {
        # "embedding_id": {"values": ["intfloat/multilingual-e5-large", "BAAI/bge-m3", "jinaai/jina-embeddings-v2-base-de"]},
        # "embedding_id": {"values": ["intfloat/multilingual-e5-large", "deepset/gbert-large", "mixedbread-ai/mxbai-embed-large-v1", "nomic-ai/nomic-embed-text-v1.5"]},
        # "embedding_id": {"values": ["intfloat/multilingual-e5-large", "deepset/gbert-large", "mixedbread-ai/mxbai-embed-large-v1"]},
        # "embedding_id": {"values": ["intfloat/multilingual-e5-large", "deepset/gbert-large", "deutsche-telekom/gbert-large-paraphrase-cosine"]},
        # "embedding_id": {"values": ["intfloat/multilingual-e5-large", "thenlper/gte-large", "BAAI/bge-large-en-v1.5"]},
        "embedding_id": {"values": ["intfloat/multilingual-e5-large", "mixedbread-ai/mxbai-embed-large-v1"]},
        # "embedding_id": {"values": ["intfloat/multilingual-e5-large"]},
        # "embedding_id": {"values": ["intfloat/multilingual-e5-small", "intfloat/multilingual-e5-base", "intfloat/multilingual-e5-large"]},
        # "chunk_size": {"value": 512},
        # "retriever_top_k": {"values": [20, 10]},
        # "retriever_top_k": {"values": [10, 5, 2, 1]},
        # "retriever_top_k": {"values": [10]},
        # "retriever_top_k": {"values": [5]},
        # "retriever_top_k": {"values": [5, 3, 1]},
        "retriever_top_k": {"values": [10, 5, 3]},
        # "retriever_top_k": {"values": [3]},
        "reranker_id": {"values": ["ColBERTv2", "BAAI/bge-reranker-large"]},
        # "reranker_id": {"values": ["BAAI/bge-reranker-large"]},
        # "reranker_id": {"values": ["ColBERTv2"]},
        "reranker_top_k": {"values": [3, 2, 1]},
        # "reranker_top_k": {"values": [2]},
        "llm_id": {"values": ["gpt-3.5-turbo", "openai/TheBloke/DiscoLM-120b-AWQ"]}
        # "llm_id": {"values": ["openai/TheBloke/DiscoLM-120b-AWQ"]}
        # "llm_id": {"values": ["gpt-3.5-turbo"]}
    },
}

EMBEDDING_MODEL_CONFIGS = {
    "intfloat/multilingual-e5-small": {
        "query_instruction": "query: ",
        "text_instruction": "passage: ",
        "pooling": "mean",
        "trust_remote_code": False,
    },
    "intfloat/multilingual-e5-base": {
        "query_instruction": "query: ",
        "text_instruction": "passage: ",
        "pooling": "mean",
        "trust_remote_code": False,
    },
    "intfloat/multilingual-e5-large": {
        "query_instruction": "query: ",
        "text_instruction": "passage: ",
        "pooling": "mean",
        "trust_remote_code": False,
    },
    "BAAI/bge-m3": {
        "query_instruction": None,
        "text_instruction": None,
        "pooling": "cls",
        "trust_remote_code": False,
    },
    "jinaai/jina-embeddings-v2-base-de": {
        "query_instruction": None,
        "text_instruction": None,
        "pooling": "mean",
        "trust_remote_code": True,
    },
    "deepset/gbert-large": {
        "query_instruction": None,
        "text_instruction": None,
        "pooling": "mean",
        "trust_remote_code": False,
    },
    "deutsche-telekom/gbert-large-paraphrase-cosine": {
        "query_instruction": None,
        "text_instruction": None,
        "pooling": "mean",
        "trust_remote_code": False,
    },
    "mixedbread-ai/mxbai-embed-large-v1": {
        "query_instruction": "Represent this sentence for searching relevant passages: ",
        "text_instruction": None,
        "pooling": "cls",
        "trust_remote_code": False,
    },
    "thenlper/gte-large": {
        "query_instruction": None,
        "text_instruction": None,
        "pooling": "mean",
        "trust_remote_code": False,
    },
    "BAAI/bge-large-en-v1.5": {
        "query_instruction": "Represent this sentence for searching relevant passages: ",
        "text_instruction": None,
        "pooling": "cls",
        "trust_remote_code": False,
    },
    # "/Users/rasdani/git/nomic-embed-text-v1.5": {
    #     "query_instruction": "search_query: ",
    #     "text_instruction": None,
    #     "pooling": "mean",
    #     "trust_remote_code": True,
    # },
}

