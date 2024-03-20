import os
import random

from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from loguru import logger
from llama_index.core import Document
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset, generate_question_context_pairs
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.llms.openrouter import OpenRouter
from transformers import AutoTokenizer
from typing import Dict, List

load_dotenv(override=True)

def load_qa_dataset(num_samples: int):
    revision = "claude-200"
    eval_dataset = load_from_hf_hub_as_embedding_qa("ellamind/sgb-embedding-qa", revision=revision, num_samples=num_samples)
    return eval_dataset

def upload_embedding_qa_to_hf_hub(embedding_qa_dataset, revision: str, dataset_repo_id: str = "ellamind/sgb-embedding-qa") -> None:
    
    """Upload an EmbeddingQAFinetuneDataset object as a dataset to the Hugging Face Hub using the push_to_hub method.

    Args:
        embedding_qa_dataset: An instance of EmbeddingQAFinetuneDataset.
        revision: A string specifying the revision name for the initial upload.
        dataset_repo_id: A string specifying the repository id for the initial upload.
    """
    # Initialize dictionaries to hold column data
    data = {
        "query_id": [],
        "query": [],
        "doc_id": [],
        "doc": [],
        "mode": []
    }

    # Populate the dictionaries with data
    for query_id, doc_ids in embedding_qa_dataset.relevant_docs.items():
        query = embedding_qa_dataset.queries[query_id]
        if "Absatz" in query:
            logger.info(f"Skipping query '{query}' because it contains the word 'Absatz'")
            continue
        for doc_id in doc_ids:
            corpus_entry = embedding_qa_dataset.corpus.get(doc_id, "")
            data["query_id"].append(query_id)
            data["query"].append(query)
            data["doc_id"].append(doc_id)
            data["doc"].append(corpus_entry)
            data["mode"].append(embedding_qa_dataset.mode)

    
    # Create a Hugging Face dataset from the list of dictionaries
    output_ds = Dataset.from_dict(data)
    
    # Push the dataset to the HF hub with the specified revision
    output_ds.push_to_hub(
        dataset_repo_id,
        split="train",
        private=True,
        revision=revision,
    )
    
    # Push the dataset to the main revision
    output_ds.push_to_hub(
        dataset_repo_id,
        split="train",
        private=True,
        revision="main",
    )

def load_from_hf_hub_as_embedding_qa(dataset_name: str, revision: str = "main", num_samples=None) -> EmbeddingQAFinetuneDataset:
    """Load a dataset from the Hugging Face Hub and return it as an EmbeddingQAFinetuneDataset object.

    Args:
        dataset_name: The name of the dataset on the Hugging Face Hub.
        revision: A string specifying the revision name for the initial upload.
        num_samples: The number of samples to load from the dataset.

    Returns:
        An instance of EmbeddingQAFinetuneDataset.
    """
    # Load the dataset from the Hugging Face Hub
    logger.info(f"Loading dataset {dataset_name} from Hugging Face Hub")
    ds = load_dataset(dataset_name, split="train", revision=revision)
    if num_samples:
        ds = ds.select(range(num_samples))

    # Initialize containers for the EmbeddingQAFinetuneDataset fields
    queries: Dict[str, str] = {}
    corpus: Dict[str, str] = {}
    relevant_docs: Dict[str, List[str]] = {}

    # Populate the containers with data from the loaded dataset
    for entry in ds:
        query_id = entry["query_id"]
        doc_id = entry["doc_id"]
        query = entry["query"]
        doc = entry["doc"]
        mode = entry.get("mode", "text")  # Assuming 'text' as default mode

        # Update queries and corpus dictionaries
        queries[query_id] = query
        corpus[doc_id] = doc

        # Update relevant_docs dictionary
        if query_id not in relevant_docs:
            relevant_docs[query_id] = [doc_id]
        else:
            relevant_docs[query_id].append(doc_id)

    # Create an EmbeddingQAFinetuneDataset object
    logger.info("Casting to LlamaIndex EmbeddingQAFinetuneDataset")
    embedding_qa_dataset = EmbeddingQAFinetuneDataset(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        mode=mode
    )

    return embedding_qa_dataset

def get_nodes(dataset_name="ellamind/sgb_dataset_cleaned", num_nodes=None, shuffle=True):
    """Load a dataset from the Hugging Face Hub and return it as a list of Node objects.
    The nodes are shuffled by default.

    Args:
        dataset_name: The name of the dataset on the Hugging Face Hub.
        num_nodes: The number of nodes to load from the dataset.
        shuffle: Whether to shuffle the nodes.

    Returns:
        A list of Node objects.
    """
    ds = load_dataset(dataset_name, split="train")

    #TODO: make embedding of metadata a hyperparamer
    docs = [Document(text=example["markdown"], metadata={"title": example["title"]}) for example in ds]


    embedding_id = "intfloat/multilingual-e5-small" # model with smallest token limit so far
    tokenizer = AutoTokenizer.from_pretrained(embedding_id)
    chunk_size = 512
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=0, tokenizer=lambda x: tokenizer(x)["input_ids"])
    nodes = splitter.get_nodes_from_documents(docs)

    # by default, the node ids are set to random uuids. To ensure same id's per run, we manually set them.
    for idx, node in enumerate(nodes):
        node.id_ = f"node_{idx}"

    if shuffle:
        random.seed("ellamind")
        random.shuffle(nodes)

    def check_truncation(nodes, tokenizer, chunk_size): # still a bit confused by how/if LLamaIndex handles truncation
        for node in nodes:
            metadata_str = ""
            for key, value in node.metadata.items():
                metadata_str += node.metadata_template.format(key=key, value=value) + node.metadata_seperator
            text = node.text_template.format(metadata_str=metadata_str, content=node.text)
            assert len(tokenizer(text)["input_ids"]) <= chunk_size

    check_truncation(nodes, tokenizer, chunk_size)

    return nodes[:num_nodes]

def get_eval_nodes(num_eval_nodes=40):
    nodes = get_nodes()
    return nodes[-num_eval_nodes:]

def generate_evaluation_dataset_and_save_to_hf_hub(revision: str, num_eval_nodes=40):
    eval_nodes = get_eval_nodes(num_eval_nodes=num_eval_nodes)
    generate_finetuning_dataset_and_save_to_hf_hub(nodes=eval_nodes, revision=revision)


def generate_finetuning_dataset_and_save_to_hf_hub(nodes, revision: str):
    llm = OpenRouter(model="anthropic/claude-3-opus", api_key=os.getenv("OPENROUTER_API_KEY"))
    # llm = OpenAI(model="gpt-4")

    qa_dataset = generate_question_context_pairs(
        nodes, llm=llm, num_questions_per_chunk=1, 
    )

    upload_embedding_qa_to_hf_hub(qa_dataset, revision=revision)

if __name__ == "__main__":
    revision = "claude-200"
    generate_evaluation_dataset_and_save_to_hf_hub(revision=revision)
