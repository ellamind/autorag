import time


from llama_index.core.evaluation import RetrieverEvaluator, FaithfulnessEvaluator
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.openai import OpenAI
from loguru import logger
import wandb

from autorag.dataset import load_qa_dataset
from autorag.retriever import get_query_engine, get_retriever
from autorag.reranker import get_node_postprocessors


# num_samples = 40
# num_samples = 10
num_samples = 3
dataset = load_qa_dataset(num_samples=num_samples)


def get_retrieval_evaluator(retriever, node_postprocessors):
    logger.info("Setting up retrieval evaluator")
    return RetrieverEvaluator.from_metric_names(
        metric_names=["mrr"], retriever=retriever, node_postprocessors=node_postprocessors
    )

def get_response_evaluator():
    # evaluator_llm = OpenAI(model="gpt-3.5-turbo")
    evaluator_llm = OpenAI(model="gpt-4-turbo-preview")

    logger.info("Setting up response evaluator")
    return FaithfulnessEvaluator(llm=evaluator_llm)

def evaluate_response(query: str, query_engine: RetrieverQueryEngine, response_evaluator: FaithfulnessEvaluator):
    logger.info(f"Query: {query}")
    response = query_engine.query(query)
    logger.info(f"Response: {response}")
    eval_result = response_evaluator.evaluate_response(response=response)
    return eval_result.score

# WIP: don't overoptimize before moving to GPU
def evaluate():
    # async def aevaluate(): # need to wrap for wandb agent
    def aevaluate():
        run = wandb.init()
        config = wandb.config

        retriever = get_retriever(config)
        node_postprocessors = get_node_postprocessors(config)
        query_engine = get_query_engine(retriever=retriever, node_postprocessors=node_postprocessors)

        retrieval_evaluator = get_retrieval_evaluator(retriever=retriever, node_postprocessors=node_postprocessors)
        response_evaluator = get_response_evaluator()

        logger.info("Evaluating...")
        start = time.time()

        # workers = 8
        # eval_results = await evaluator.aevaluate_dataset(dataset, workers=workers)

        eval_results = {"MRR": [], "faithfulness": []}
        for i, (query_id, query) in enumerate(dataset.queries.items()):
            sample_expected = dataset.relevant_docs[query_id]

            retrieval_eval_result = retrieval_evaluator.evaluate(query, sample_expected)
            logger.info(f"Retrieved texts: {retrieval_eval_result.retrieved_texts}")
            
            logger.info(f"LLM: {config['llm_id']}")
            faithfulness_score = evaluate_response(query, query_engine, response_evaluator) # TODO: look into making this async

            eval_results["MRR"].append(retrieval_eval_result.metric_dict["mrr"].score)
            eval_results["faithfulness"].append(faithfulness_score)
            logger.info(f"Evaluated query {i+1}/{len(dataset.queries)}")


        avg_mrr = sum(eval_results["MRR"]) / len(eval_results["MRR"])
        avg_faithfulness = sum(eval_results["faithfulness"]) / len(eval_results["faithfulness"])

        stop = time.time()
        logger.info(f"Evaluation took {stop-start} seconds")

        # wandb.log({"MRR": avg_mrr, "faithfulness": avg_faithfulness})
        wandb.log({"MRR": avg_mrr})
        wandb.log({"faithfulness": avg_faithfulness})

        run.finish()

    # asyncio.run(aevaluate()) # wandb agent does not accept coroutines
    aevaluate()
